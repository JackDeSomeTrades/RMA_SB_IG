from logging import warning
from isaacgym import gymapi
import torch
from rma_sb_ig.utils.terrain import Terrain
import os
import sys
from rma_sb_ig.utils.helpers import *
import xml.etree.ElementTree as ET
import warnings
from isaacgym import gymutil
from isaacgym.torch_utils import *
import math


class SotoEnvScene:
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.gym = gymapi.acquire_gym()
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        self.is_test_mode = self.cfg.sim_param.test
        sim_device, self.sim_device_id = gymutil.parse_device_str(
            self.sim_device)
        self.headless = headless
        self.num_envs = cfg.env.num_envs

        self.device = sim_device if self.sim_params.use_gpu_pipeline else 'cpu'
        self.graphics_device_id = self.sim_device_id
        # configure sim
        self._adjust_sim_param()

        # create sim
        self.enable_viewer_sync = True
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        if not self.headless:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                print("*** Failed to create viewer")
                quit()
        else :
            self.viewer = None

        self.create_sim()
        if not self.headless :
            self._define_viewer()
        # point camera at middle env

        self.gym.prepare_sim(self.sim)  # TODO : ATTENTION FAIT TOUT PLANTER avec set actor dof state


    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        # Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = 0
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        # used to control the elasticity of collisions with the ground plane
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

        self._create_assets()

    def _adjust_sim_param(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        self.sim_params.dt = self.cfg.sim_param.dt
        self.sim_params.substeps = self.cfg.sim_param.substep
        self.sim_params.use_gpu_pipeline = self.cfg.sim_param.use_gpu
        if self.physics_engine == gymapi.SIM_PHYSX:
            self.sim_params.physx.num_position_iterations = self.cfg.sim_param.num_position_iterations
            self.sim_params.physx.num_velocity_iterations = self.cfg.sim_param.num_velocity_iterations
            self.sim_params.physx.num_threads = self.cfg.sim_param.num_threads
            self.sim_params.physx.use_gpu = self.cfg.sim_param.use_gpu_physx
        else:
            raise Exception("This robot can only be used with PhysX")

    def _create_assets(self):

        asset_path = self.cfg.asset.file.format(ROOT_DIR=get_project_root())
        asset_root = os.path.dirname(asset_path)

        box_limits = (self.cfg.domain_rand.length_box,
                      self.cfg.domain_rand.width_box,
                      self.cfg.domain_rand.height_box)

        asset_options = gymapi.AssetOptions()
        asset_options.use_mesh_materials = True
        # load soto asset
        asset_file = os.path.basename(asset_path)
        asset_options.armature = self.cfg.asset.armature

        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.density = 10
        self.box_dimensions = [self._get_random_boxes(*box_limits) for i in range(self.num_envs)]
        self.l_boxes_asset = [self.gym.create_box(self.sim, *dim, asset_options) for dim in self.box_dimensions]
        asset_options.disable_gravity = self.cfg.asset.disable_gravity
        asset_options.override_com = self.cfg.asset.override_com
        asset_options.vhacd_enabled = True
        asset_options.override_inertia = self.cfg.asset.override_inertia
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        self.soto_asset = self.gym.load_asset(self.sim, asset_root,asset_file,asset_options)
        # # configure soto dofs
        self.dof_names = self.gym.get_asset_dof_names(self.soto_asset)
        # # self.default_dof_pos = np.zeros(self.soto_num_dofs, dtype=np.float32) #way to initialize dofs
        self.num_dofs = self.gym.get_asset_dof_count(self.soto_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.soto_asset)
        # dof_props_asset = self.gym.get_asset_dof_properties(self.soto_asset)

        self.soto_dof_props = self.gym.get_asset_dof_properties(self.soto_asset)
        self.lower_bounds_joints = self.soto_dof_props["lower"]
        self.upper_bounds_joints = self.soto_dof_props["upper"]

        self.motor_strength = self.soto_dof_props["effort"]
        self.joint_velocity = self.soto_dof_props["velocity"]
        self.soto_mids = 0.5*(self.upper_bounds_joints +self.lower_bounds_joints)
        self.default_dof_pos = self.soto_mids
        # remember : important pieces to control are conveyor belt left base link/conveyor belt right base link

        self.default_dof_state = np.zeros(
            self.num_dofs, gymapi.DofState.dtype)

        self.default_dof_state["pos"] = self.default_dof_pos

        # get link index of soto pieces, which we will use as effectors
        self.soto_link_dict = self.gym.get_asset_rigid_body_dict(
            self.soto_asset)
        self.index_to_get = ['gripper_y_right_link', 'gripper_y_left_link']

        self.soto_indexs = [self.soto_link_dict[i] for i in self.index_to_get]
        self.soto_pose = gymapi.Transform()
        self.soto_pose.p = gymapi.Vec3(*self.cfg.init_state.pos)
        self.box_pose = gymapi.Transform()

        self.body_names = self.gym.get_asset_rigid_body_names(self.soto_asset)
        self.right_conv_belt_id = self.dof_names.index("conveyor_right_to_belt")
        self.left_conv_belt_id = self.dof_names.index("conveyor_left_to_belt")
        self.feet_names = [s for s in self.body_names if self.cfg.asset.foot_name in s]

        self.penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            self.penalized_contact_names.extend(
                [s for s in self.body_names if name in s])
        self.termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            self.termination_contact_names.extend(
                [s for s in self.body_names if name in s])

        self._initialize_env()

    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)

    def _initialize_env(self):
        # configure env grid
        self.num_per_row = int(np.sqrt(self.num_envs))
        spacing = self.cfg.terrain.border_size
        self.env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        self.env_upper = gymapi.Vec3(spacing, spacing, spacing)

        print("Creating %d environments" % self.num_envs)
        self.envs = []
        self.box_masses = []
        self.box_com_x = []
        self.box_com_y = []
        self.box_com_z = []
        self.actor_handles = []
        self.box_handles = []
        self.soto_fric = []
        self.box_fric = []

        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(
                self.sim, self.env_lower, self.env_upper, self.num_per_row)

            # get soto_id in environnement(always the same )
            self.soto_handle = self.gym.create_actor(
                env, self.soto_asset, self.soto_pose, self.cfg.asset.name, i,
                1,0)

            index_rotate = self.gym.find_actor_dof_index(env, self.soto_handle, "gripper_rotate", gymapi.DOMAIN_ENV)
            index_x = self.gym.find_actor_dof_index(env, self.soto_handle, "gripper_base_x", gymapi.DOMAIN_ENV)
            self.default_dof_state["pos"][index_rotate] = self.cfg.init_state.angle
            self.default_dof_state["pos"][index_x] = self.lower_bounds_joints[index_x]
            # set dof properties
            self.gym.set_actor_dof_properties(
                env, self.soto_handle, self.soto_dof_props)

            soto_properties = self.gym.get_actor_rigid_shape_properties(env,self.soto_handle)
            soto_properties = self._process_rigid_properties(soto_properties, "soto")
            self.gym.set_actor_rigid_shape_properties(env, self.soto_handle, soto_properties)
            # set initial dof states
            self.gym.set_actor_dof_states(
                env, self.soto_handle, self.default_dof_state, gymapi.DOMAIN_ENV)

            # add box
            self.gripper_x_id = self.gym.find_actor_rigid_body_index(
                env, self.soto_handle, "gripper_base_x_link", gymapi.DOMAIN_ENV)

            self.conveyor_left_id = self.gym.find_actor_rigid_body_index(
                env, self.soto_handle, "gripper_y_left_link", gymapi.DOMAIN_ENV)
            self.conveyor_right_id = self.gym.find_actor_rigid_body_index(
                env, self.soto_handle, "gripper_y_right_link", gymapi.DOMAIN_ENV)
            body_states = self.gym.get_actor_rigid_body_states(env, self.soto_handle, gymapi.DOMAIN_ENV)

            self.box_pose.p.x = body_states["pose"][self.gripper_x_id][0]["x"]
            self.box_pose.p.y = body_states["pose"][self.gripper_x_id][0]["y"]
            self.box_pose.p.z = body_states["pose"][self.gripper_x_id][0]["z"] + self.box_dimensions[i][2]/2+0.132


            vec_add = gymapi.Vec3(-self.box_dimensions[i][0]/2+0.15,0.0,0.0)

            quat = gymapi.Quat.from_axis_angle(gymapi.Vec3(0.,0.,1.),self.default_dof_state["pos"][index_rotate])

            self.box_pose.p +=quat.rotate(vec_add)
            r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0.,0.,1.),np.pi)

            self.box_pose.r = gymapi.Quat(*body_states["pose"][self.gripper_x_id][1])*r
            self.box_handle = self.gym.create_actor(env, self.l_boxes_asset[i], self.box_pose, "box", i,0,1)
            color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
            self.gym.set_rigid_body_color(env, self.box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

            box_shape_props = self.gym.get_actor_rigid_shape_properties(env, self.box_handle)
            box_shape_props = self._process_rigid_properties(box_shape_props, "box")

            self.gym.set_actor_rigid_shape_properties(env, self.box_handle, box_shape_props)
            box_properties = self.gym.get_actor_rigid_body_properties(env, self.box_handle)
            box_properties = self._process_box_props(box_properties)
            self.gym.set_actor_rigid_body_properties(
                env, self.box_handle, box_properties)
            
            # get box id (always the same at each iteration)
            self.box_idx = self.gym.get_actor_rigid_body_index(
                env, self.box_handle, 0, gymapi.DOMAIN_SIM)
            self.actor_handles.append(self.soto_handle)
            self.box_handles.append(self.box_handle)

            self.envs.append(env)

        self.box_masses = torch.cuda.FloatTensor(self.box_masses)
        self.box_com_x = torch.cuda.FloatTensor(self.box_com_x)
        self.box_com_y = torch.cuda.FloatTensor(self.box_com_y)
        self.box_com_z = torch.cuda.FloatTensor(self.box_com_z)
        self.box_dim = torch.cuda.FloatTensor(self.box_dimensions)
        self.soto_fric = torch.cuda.FloatTensor(self.soto_fric)
        self.box_fric = torch.cuda.FloatTensor(self.box_fric)

        self.feet_indices = torch.zeros(
            len(self.feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], self.feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(
            self.penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        
        for i in range(len(self.penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], self.penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(
            self.termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], self.termination_contact_names[i])
        
        
        self._create_distance_sensors()

    def _process_rigid_properties(self, props,type):
        if self.cfg.domain_rand.randomize_friction:
            rng_static = self.cfg.domain_rand.friction_static_range
            rng_dynamic = self.cfg.domain_rand.friction_dynamic_range
            friction = np.random.uniform(rng_static[0], rng_static[1])
            dyn_friction = np.random.uniform(rng_dynamic[0], rng_dynamic[1])
            for i in range(len(props)) :
                props[i].compliance = 0.0
                props[i].friction = friction
                props[i].rolling_friction = dyn_friction
                props[i].restitution = self.cfg.asset.restitution
                #props[i].thickness = 0.05
                #props[i].torsion_friction  = 0.4
        if type == "soto" :
            self.soto_fric.append([friction,dyn_friction])
        elif type == "box" :
            self.box_fric.append([friction,dyn_friction])
        return props

    def _process_box_props(self,props):
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.mass_box
            props[0].mass = np.random.uniform(rng[0], rng[1])
        if self.cfg.domain_rand.randomize_com:
            rng_2 = self.cfg.domain_rand.com_distribution_range
            props[0].com.x += np.random.uniform(rng_2[0], rng_2[1])
            props[0].com.y += np.random.uniform(rng_2[0], rng_2[1])
        self.box_masses.append(props[0].mass)
        self.box_com_x.append(props[0].com.x)
        self.box_com_y.append(props[0].com.y)
        self.box_com_z.append(props[0].com.z)
        return props

    def _create_distance_sensors(self):
        self.distance_sensors = torch.zeros(self.num_envs,2,device = self.device)
        self.box_init_axis = torch.tensor([[[1,0,0],[0,1,0],[0,0,1]]],device = self.device, dtype = torch.float32).expand(self.num_envs,-1,-1)
        self.box_axis = torch.zeros_like(self.box_init_axis)
        self.gripper_init_x_axis = torch.tensor([[1,0,0]],device = self.device, dtype = torch.float32).expand(self.num_envs,-1)

    def _define_viewer(self):
        self.cam_pos = gymapi.Vec3(4, 3, 2)
        self.cam_target = gymapi.Vec3(-4, -3, 0)
        self.middle_env = self.envs[self.num_envs // 2 + self.num_per_row // 2]
        self.gym.viewer_camera_look_at(
            self.viewer, self.middle_env, self.cam_pos, self.cam_target)

    def _get_random_boxes(self, l_limit, w_limit, h_limit):
        length = random.uniform(l_limit[0], l_limit[1])
        width = random.uniform(w_limit[0], w_limit[1])
        height = random.uniform(h_limit[0], h_limit[1])
        return [length, width, height]
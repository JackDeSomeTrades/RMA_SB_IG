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
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless
        self.num_envs = cfg.env.num_envs

        if sim_device_type == 'cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'
        # configure sim
        self.graphics_device_id = self.sim_device_id

        if self.headless == True:
            self.graphics_device_id = -1
        self._adjust_sim_param()
        self.up_axis_idx = set_sim_params_up_axis(self.sim_params, 'z')
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.create_sim()
        self.gym.prepare_sim(self.sim)
        # create sim
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

        self._create_envs()

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

    def set_camera(self):
        cam_pos = gymapi.Vec3(4, 3, 2)
        cam_target = gymapi.Vec3(-4, -3, 0)
        middle_env = self.envs[self.num_envs // 2 + self.num_per_row // 2]
        self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)

    def _adjust_sim_param(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        self.sim_params.dt = self.cfg.sim_param.dt
        self.sim_params.substeps = self.cfg.sim_param.substeps
        self.sim_params.use_gpu_pipeline = self.cfg.sim_param.use_gpu
        self.sim_params.enable_actor_creation_warning = False
        if self.physics_engine == gymapi.SIM_PHYSX:
            self.sim_params.physx.bounce_threshold_velocity = 1.0*9.81*self.sim_params.dt/self.sim_params.substeps
            self.sim_params.physx.num_position_iterations = self.cfg.sim_param.physx.num_position_iterations
            self.sim_params.physx.num_velocity_iterations = self.cfg.sim_param.physx.num_velocity_iterations
            self.sim_params.physx.num_threads = self.cfg.sim_param.physx.num_threads
            self.sim_params.physx.use_gpu = self.cfg.sim_param.use_gpu
            self.sim_params.physx.max_gpu_contact_pairs = eval(self.cfg.sim_param.physx.max_gpu_contact_pairs)
            self.sim_params.physx.max_depenetration_velocity = self.cfg.sim_param.physx.max_depenetration_velocity
            self.sim_params.physx.solver_type = self.cfg.sim_param.physx.solver_type
        else :
            raise Exception("This robot can only be used with PhysX")

    def _get_soto_asset_option(self,asset_options):
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.disable_gravity = self.cfg.asset.disable_gravity
        asset_options.override_com = self.cfg.asset.override_com
        asset_options.vhacd_enabled = True
        asset_options.override_inertia = self.cfg.asset.override_inertia
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        return asset_options

    def _get_box_asset_option(self,asset_options):
        asset_options.density = 10
        asset_options.override_inertia = self.cfg.asset.override_inertia
        return asset_options

    def _create_envs(self):

        asset_path = self.cfg.asset.file.format(ROOT_DIR=get_project_root())
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        soto_asset_options = gymapi.AssetOptions()
        box_asset_options = gymapi.AssetOptions()
        box_limits = (self.cfg.domain_rand.length_box,
                      self.cfg.domain_rand.width_box,
                      self.cfg.domain_rand.height_box)

        soto_asset_options = self._get_soto_asset_option(soto_asset_options)
        box_asset_options = self._get_box_asset_option(box_asset_options)

        self.box_dimensions = [self._get_random_boxes(*box_limits) for _ in range(self.num_envs)]

        self.l_boxes_asset = [self.gym.create_box(self.sim, *dim, box_asset_options) for dim in self.box_dimensions]
        self.soto_asset = self.gym.load_asset(self.sim, asset_root, asset_file, soto_asset_options)

        self.num_dofs = self.gym.get_asset_dof_count(self.soto_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.soto_asset)
        soto_dof_props = self.gym.get_asset_dof_properties(self.soto_asset)
        self._get_properties(soto_dof_props)

        self.body_names = self.gym.get_asset_rigid_body_names(self.soto_asset)
        self.dof_names = self.gym.get_asset_dof_names(self.soto_asset)
        self.penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            self.penalized_contact_names.extend(
                [s for s in self.body_names if name in s])
        self.termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            self.termination_contact_names.extend(
                [s for s in self.body_names if name in s])

        soto_pose = gymapi.Transform()
        soto_pose.p = gymapi.Vec3(*self.cfg.init_state.pos)
        box_pose = gymapi.Transform()
        default_dof_pos = self._process_dof_pos()
        default_dof_state = np.zeros(self.num_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] = default_dof_pos
        self._find_ids()


        print("Creating %d environments" % self.num_envs)
        self._configure_env_grid()
        self.actor_handles = []
        self.envs = []
        self.box_masses = []
        self.box_com_x = []
        self.box_com_y = []
        self.box_com_z = []
        self.box_handles = []
        self.soto_fric = []
        self.box_fric = []

        for i in range(self.num_envs):
            # create env
            color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
            env = self.gym.create_env(self.sim, self.env_lower, self.env_upper, self.num_per_row)

            rigid_soto_properties = self.gym.get_asset_rigid_shape_properties(self.soto_asset)
            rigid_soto_properties = self._process_rigid_properties(rigid_soto_properties, "soto")
            self.gym.set_asset_rigid_shape_properties(self.soto_asset, rigid_soto_properties)
            soto_handle = self.gym.create_actor(env, self.soto_asset, soto_pose, self.cfg.asset.name, i, 1, 0)
            self.gym.set_actor_dof_properties(env, soto_handle, soto_dof_props)
            self.gym.set_actor_dof_states(env, soto_handle, default_dof_state, gymapi.DOMAIN_ENV)

            body_states = self.gym.get_actor_rigid_body_states(env, soto_handle, gymapi.DOMAIN_ENV)
            box_pose = self._process_box_pos(box_pose,default_dof_state,body_states,i)
            self.box_handle = self.gym.create_actor(env, self.l_boxes_asset[i], box_pose, "box", i, 0, 1)
            self.gym.set_rigid_body_color(env, self.box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            box_shape_props = self.gym.get_actor_rigid_shape_properties(env, self.box_handle)
            box_shape_props = self._process_rigid_properties(box_shape_props, "box")
            self.gym.set_actor_rigid_shape_properties(env, self.box_handle, box_shape_props)
            box_properties = self.gym.get_actor_rigid_body_properties(env, self.box_handle)
            box_properties = self._process_box_props(box_properties)
            self.gym.set_actor_rigid_body_properties(env, self.box_handle, box_properties)

            self.actor_handles.append(soto_handle)
            self.box_handles.append(self.box_handle)
            self.envs.append(env)

        self.box_masses = torch.cuda.FloatTensor(self.box_masses)
        self.box_com_x = torch.cuda.FloatTensor(self.box_com_x)
        self.box_com_y = torch.cuda.FloatTensor(self.box_com_y)
        self.box_com_z = torch.cuda.FloatTensor(self.box_com_z)
        self.box_dim = torch.cuda.FloatTensor(self.box_dimensions)
        self.soto_fric = torch.cuda.FloatTensor(self.soto_fric)
        self.box_fric = torch.cuda.FloatTensor(self.box_fric)

        self.penalised_contact_indices = torch.zeros(len(self.penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], self.penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(self.termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], self.termination_contact_names[i])

        self._create_distance_sensors()


    def _process_box_pos(self,box_pose,default_dof_state,body_states,i):
        vec_add = gymapi.Vec3(-self.box_dimensions[i][0] / 2 + 0.15, 0.0, 0.0)
        quat = gymapi.Quat.from_axis_angle(gymapi.Vec3(0., 0., 1.), default_dof_state["pos"][self.index_rotate])
        r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0., 0., 1.), np.pi)
        box_pose.p.x = body_states["pose"][self.gripper_x_id][0]["x"]
        box_pose.p.y = body_states["pose"][self.gripper_x_id][0]["y"]
        box_pose.p.z = body_states["pose"][self.gripper_x_id][0]["z"] + self.box_dimensions[i][2] / 2 + 0.142
        box_pose.p += quat.rotate(vec_add)
        box_pose.r = gymapi.Quat(*body_states["pose"][self.gripper_x_id][1]) * r
        return box_pose

    def _find_ids(self):
        self.gripper_x_id = self.gym.find_asset_rigid_body_index(self.soto_asset, "gripper_base_x_link")
        self.right_conv_belt_id = self.dof_names.index("conveyor_right_to_belt")
        self.left_conv_belt_id = self.dof_names.index("conveyor_left_to_belt")
        self.conveyor_left_id = self.gym.find_asset_rigid_body_index(self.soto_asset, "gripper_y_left_link")
        self.conveyor_right_id = self.gym.find_asset_rigid_body_index(self.soto_asset, "gripper_y_right_link")

    def _configure_env_grid(self):
        self.num_per_row = int(np.sqrt(self.num_envs))
        spacing = self.cfg.terrain.border_size
        self.env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        self.env_upper = gymapi.Vec3(spacing, spacing, spacing)

    def _process_dof_pos(self):
        default_dof_pos = 0.5 * (self.upper_bounds_joints + self.lower_bounds_joints)
        k = 0.25
        cbl = self.dof_names.index("gripper_y_left")
        cbr = self.dof_names.index("gripper_y_right")
        default_dof_pos[cbl] = (1 - k) * self.lower_bounds_joints[cbl] + k * self.upper_bounds_joints[cbl]
        default_dof_pos[cbr] = (1 - k) * self.lower_bounds_joints[cbr] + k * self.upper_bounds_joints[cbr]
        self.index_rotate = self.gym.find_asset_dof_index(self.soto_asset, "gripper_rotate")
        index_x = self.gym.find_asset_dof_index(self.soto_asset, "gripper_base_x")
        default_dof_pos[self.index_rotate] = self.cfg.init_state.angle
        default_dof_pos[index_x] = self.lower_bounds_joints[index_x]
        self.default_dof_pos = torch.tensor(default_dof_pos[:], device = self.device).expand(self.num_envs,self.num_dofs)
        return default_dof_pos

    def _get_properties(self,soto_dof_props):
        self.lower_bounds_joints = soto_dof_props["lower"]
        self.upper_bounds_joints = soto_dof_props["upper"]
        self.motor_strength = soto_dof_props["effort"]
        self.joint_velocity = soto_dof_props["velocity"]


    def _process_rigid_properties(self, props, type ):
        friction = 0.8
        if self.cfg.domain_rand.randomize_friction:
            rng_static = self.cfg.domain_rand.friction_static_range
            friction = np.random.uniform(rng_static[0], rng_static[1])
            for i in range(len(props)) :
                props[i].compliance = 0.0
                props[i].friction = friction
                #props[i].rolling_friction = dyn_friction
                props[i].restitution = self.cfg.asset.restitution
                #props[i].thickness = 1.0
                #props[i].torsion_friction  = 0.4
        if type == "soto" :
            self.soto_fric.append([friction])
        elif type == "box" :
            self.box_fric.append([friction])
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


    def _get_random_boxes(self, l_limit, w_limit, h_limit):
        length = random.uniform(l_limit[0], l_limit[1])
        width = random.uniform(w_limit[0], w_limit[1])
        height = random.uniform(h_limit[0], h_limit[1])
        return [length, width, height]
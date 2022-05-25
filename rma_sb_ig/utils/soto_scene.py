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
        if self.headless:
            self.graphics_device_id = -1
        # configure sim
        self._adjust_sim_param()

        # create sim
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.create_sim()

        # point camera at middle env
        if not self.headless:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
        self._define_viewer()

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        # Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = 0
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution  # used to control the elasticity of collisions with the ground plane
        self.gym.add_ground(self.sim, plane_params)

        self._create_assets()

    def _adjust_sim_param(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        self.sim_params.dt = self.cfg.sim_param.dt
        self.sim_params.substeps = self.cfg.sim_param.substep
        self.sim_params.use_gpu_pipeline = self.cfg.sim_param.use_gpu
        if self.physics_engine == gymapi.SIM_PHYSX:
            self.sim_params.physx.solver_type = self.cfg.sim_param.solver_type
            self.sim_params.physx.num_position_iterations = self.cfg.sim_param.num_position_iterations
            self.sim_params.physx.num_velocity_iterations = self.cfg.sim_param.num_velocity_iterations
            self.sim_params.physx.rest_offset = self.cfg.sim_param.rest_offset
            self.sim_params.physx.contact_offset = self.cfg.sim_param.contact_offset
            self.sim_params.physx.friction_offset_threshold = self.cfg.sim_param.friction_offset_threshold
            self.sim_params.physx.friction_correlation_distance = self.cfg.sim_param.friction_correlation_distance
            self.sim_params.physx.num_threads = self.cfg.sim_param.num_threads
            self.sim_params.physx.use_gpu = self.cfg.sim_param.use_gpu_physx
        else:
            raise Exception("This robot can only be used with PhysX")

    def _create_assets(self):

        asset_path = self.cfg.asset.file.format(ROOT_DIR=get_project_root())
        asset_root = os.path.dirname(asset_path)

        box_limits = (self.cfg.box.lim_x,
                      self.cfg.box.lim_y,
                      self.cfg.box.lim_z)

        asset_options = gymapi.AssetOptions()

        self.l_boxes_asset = [self.gym.create_box(self.sim, *self._get_random_boxes(
            *box_limits), asset_options) for i in range(self.num_envs)]

        # load soto asset

        asset_file = os.path.basename(asset_path)

        asset_options.armature = self.cfg.asset.armature
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.disable_gravity = self.cfg.asset.disable_gravity
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.override_com = self.cfg.asset.override_com
        asset_options.override_inertia = self.cfg.asset.override_inertia

        self.soto_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options)

        # # configure soto dofs

        # # self.default_dof_pos = np.zeros(self.soto_num_dofs, dtype=np.float32) #way to initialize dofs
        self.num_dofs = self.gym.get_asset_dof_count(self.soto_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.soto_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(self.soto_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(
            self.soto_asset)

        self.soto_dof_props = self.gym.get_asset_dof_properties(
            self.soto_asset)
        self.soto_lower_limits = self.soto_dof_props["lower"]
        self.soto_upper_limits = self.soto_dof_props["upper"]
        self.soto_motor_strength = self.soto_dof_props["effort"]
        self.soto_joint_velocity = self.soto_dof_props["velocity"]
        self.soto_mids = 0.5 * \
                         (self.soto_upper_limits + self.soto_lower_limits)

        self.default_dof_pos = self.soto_mids
        # remember : important pieces to control are conveyor belt left base link/conveyor belt right base link

        self.default_dof_state = np.zeros(
            self.num_dofs, gymapi.DofState.dtype)
        self.default_dof_state["pos"] = self.default_dof_pos
        #properties of distance_sensor
        self._define_distance_sensor()

        # send to torch
        self.default_dof_pos_tensor = to_torch(
            self.default_dof_pos, device=self.device)
        ## get link index of soto pieces, which we will use as effectors
        # vertical movment : vertical axis link
        # Z rotate mov = gripper_base_link
        # lateral translation : gripper_base_x_link
        # space beetween grippers : gripper_y_left_link/gripper_y_right_link
        self.soto_link_dict = self.gym.get_asset_rigid_body_dict(self.soto_asset)
        self.index_to_get = ['gripper_y_right_link','gripper_y_left_link']
        self.soto_indexs = [self.soto_link_dict[i] for i in self.index_to_get]
        print(self.soto_indexs)
        self.soto_pose = gymapi.Transform()
        self.soto_pose.p = gymapi.Vec3(*self.cfg.init_state.pos)
        self.box_pose = gymapi.Transform()
        self._initialize_env()

    def _initialize_env(self):
        # configure env grid
        self.num_per_row = int(np.sqrt(self.num_envs))
        spacing = self.cfg.terrain.border_size
        self.env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        self.env_upper = gymapi.Vec3(spacing, spacing, spacing)
        print("Creating %d environments" % self.num_envs)
        self.envs = []
        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(
                self.sim, self.env_lower, self.env_upper, self.num_per_row)
            # add box
            self.box_pose.p.x = np.random.uniform(-0.1, 0.1)
            self.box_pose.p.y = np.random.uniform(-0.1, 0.1)
            self.box_pose.p.z = 0.5
            self.box_pose.r = gymapi.Quat.from_axis_angle(
                gymapi.Vec3(0, 0, 1), np.random.uniform(-0.2, 0.2))

            self.box_handle = self.gym.create_actor(
                env, self.l_boxes_asset[i], self.box_pose, "box", i,
                0)
            color = gymapi.Vec3(np.random.uniform(
                0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
            self.gym.set_rigid_body_color(
                env, self.box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

            # get box id (always the same at each iteration)
            self.box_idx = self.gym.get_actor_rigid_body_index(
                env, self.box_handle, 0, gymapi.DOMAIN_SIM)

            # get soto_id in environnement(always the same aswell)
            self.soto_handle = self.gym.create_actor(
                env, self.soto_asset, self.soto_pose, self.cfg.asset.name, i,
                1)
            # set dof properties
            self.gym.set_actor_dof_properties(
                env, self.soto_handle, self.soto_dof_props)

            # set initial dof states
            self.gym.set_actor_dof_states(
                env, self.soto_handle, self.default_dof_state, gymapi.STATE_ALL)

            # set initial position targets
            self.gym.set_actor_dof_position_targets(
                env, self.soto_handle, self.default_dof_pos)
            self.envs.append(env)
            # get index of pieces in rigid body state tensor
            dist1_idx = self.gym.find_actor_rigid_body_index(
                env, self.soto_handle, "gripper_y_right_to_dist_x_sensor", gymapi.DOMAIN_SIM)
            dist2_idx = self.gym.find_actor_rigid_body_index(
                env, self.soto_handle, "gripper_y_left_to_dist_x_sensor", gymapi.DOMAIN_SIM)
            # set distance sensors

            self.distance1_handle = self.gym.create_camera_sensor(env, self.distance_sensor)
            self.distance2_handle = self.gym.create_camera_sensor(env, self.distance_sensor)


            self.gym.attach_camera_to_body(self.distance1_handle, env, dist1_idx, self.local_transform, gymapi.FOLLOW_TRANSFORM)
            self.gym.attach_camera_to_body(self.distance2_handle, env, dist2_idx, self.local_transform, gymapi.FOLLOW_TRANSFORM)

    def _define_distance_sensor(self):
        self.distance_sensor = gymapi.CameraProperties()
        self.distance_sensor.width = 1
        self.distance_sensor.height = 1
        self.distance_sensor.horizontal_fov = self.cfg.distance_sensor.fov # field of view in radians (cone)
        self.distance_sensor.near_plane = self.cfg.distance_sensor.near_plane
        self.distance_sensor.far_plane = self.cfg.distance_sensor.far_plane
        self.distance_sensor.use_collision_geometry = self.cfg.distance_sensor.use_collision_geometry
        self.distance_sensor.enable_tensors = self.cfg.distance_sensor.enable_tensors

        self.local_transform = gymapi.Transform()
        self.local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.radians(45.0))
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
        return length, width, height

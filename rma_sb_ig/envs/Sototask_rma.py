from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym import gymapi
from isaacgym.torch_utils import *
import gym.spaces as gymspace
import math

from rma_sb_ig.utils.helpers import *
from rma_sb_ig.envs.forward_task import ForwardTask



class SotoRobotTask():
    def __init__(self, *args):
        #super(SotoRobotTask, self).__init__(*args)

        self.gym = gymapi.acquire_gym()

        custom_parameters = [
            {"name": "--controller", "type": str, "default": "ik",
             "help": "Controller to use for Franka. Options are {ik, osc}"},
            {"name": "--num_envs", "type": int, "default": 50, "help": "Number of environments to create"},
        ]

        self.args = gymutil.parse_arguments(
            description="Soto_gripper",
            custom_parameters=custom_parameters,
        )

        self.num_envs = self.args.num_envs

        device = self.args.sim_device if self.args.use_gpu_pipeline else 'cpu'

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        sim_params.use_gpu_pipeline = self.args.use_gpu_pipeline

        if self.args.physics_engine == gymapi.SIM_PHYSX:
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 8
            sim_params.physx.num_velocity_iterations = 1
            sim_params.physx.rest_offset = 0.0
            sim_params.physx.contact_offset = 0.001
            sim_params.physx.friction_offset_threshold = 0.001
            sim_params.physx.friction_correlation_distance = 0.0005
            sim_params.physx.num_threads = self.args.num_threads
            sim_params.physx.use_gpu = self.args.use_gpu
        else:
            raise Exception("This robot can only be used with PhysX")

        # create sim
        self.sim = self.gym.create_sim(self.args.compute_device_id, self.args.graphics_device_id, self.args.physics_engine, sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")

        # create viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise Exception("Failed to create viewer")

        asset_root = "resources/robots/"

        box_limits = ([0.1,0.5],
                  [0.1,0.5],
                  [0.1,0.4])

        self.asset_options = gymapi.AssetOptions()
        l_boxes_asset = [self.gym.create_box(self.sim, *self._get_random_boxes(*box_limits), self.asset_options) for i in range(self.num_envs)]
        
        # load soto asset
        soto_asset_file = "soto_gripper/soto_gripper.urdf"
        self.asset_options.armature = 0.01
        self.asset_options.fix_base_link = True
        self.asset_options.disable_gravity = False
        self.asset_options.flip_visual_attachments = False
        self.soto_asset = self.gym.load_asset(self.sim, asset_root, soto_asset_file, self.asset_options)

        # configure soto dofs
        self.soto_dof_props = self.gym.get_asset_dof_properties(self.soto_asset)
        self.soto_lower_limits = self.soto_dof_props["lower"]
        self.soto_upper_limits = self.soto_dof_props["upper"]
        self.soto_mids = 0.5 * (self.soto_upper_limits + self.soto_lower_limits)

        # default dof states
        self.soto_num_dofs = self.gym.get_asset_dof_count(self.soto_asset)
        #self.default_dof_pos = np.zeros(self.soto_num_dofs, dtype=np.float32) #way to initialize dofs
        self.default_dof_pos = self.soto_mids

        #remember : important pieces to control are conveyor belt left base link/conveyor belt right base link

        self.default_dof_state = np.zeros(self.soto_num_dofs, gymapi.DofState.dtype)
        self.default_dof_state["pos"] = self.default_dof_pos

        # send to torch
        self.default_dof_pos_tensor = to_torch(self.default_dof_pos, device=device)

        # get link index of soto pieces, which we will use as effectors
        # vertical movment : vertical axis link
        # Z rotate mov = gripper_base_link
        # lateral translation : gripper_base_x_link
        # space beetween grippers : gripper_y_left_link/gripper_y_right_link

        self.soto_link_dict = self.gym.get_asset_rigid_body_dict(self.soto_asset)
        index_to_get = ["vertical_axis_link","gripper_base_link","gripper_base_x_link","gripper_y_left_link","gripper_y_right_link"]

        self.soto_indexs = [self.soto_link_dict[i] for i in index_to_get]
        # self.soto_vert_ax_index = self.soto_link_dict["vertical_axis_link"]
        # self.soto_base_index = self.soto_link_dict["gripper_base_link"]
        # self.soto_base_x_index = self.soto_link_dict["gripper_base_x_link"]
        # self.soto_gripper_left_index = self.soto_link_dict["gripper_y_left_link"]
        # self.soto_gripper_right_index = self.soto_link_dict["gripper_y_right_link"]

        # configure env grid
        self.num_envs = self.args.num_envs
        self.num_per_row = int(math.sqrt(self.num_envs))
        spacing = 1.5
        self.env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        self.env_upper = gymapi.Vec3(spacing, spacing, spacing)
        print("Creating %d environments" % self.num_envs)

        self.soto_pose = gymapi.Transform()
        self.soto_pose.p = gymapi.Vec3(0, 0, 0)

        self.box_pose = gymapi.Transform()

        self.envs = []

        #global index list of soto_pieces

        self.global_soto_indexs = []
        self.box_idxs = []
        # self.vert_idx = []
        # self.base_idxs = []
        # self.base_x_idx = []
        # self.gripper_left = []
        # self.gripper_right = []

        #self.init_pos_list = [] #l_pos of a piece
        #self.init_rot_list = [] #l_rot of a piece
        self.l_handle = []
        # add ground plane
        self.plane_params = gymapi.PlaneParams()
        self.plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.plane_params.distance = 0
        self.plane_params.dynamic_friction = 1.0
        self.plane_params.static_friction = 1.0
        self.gym.add_ground(self.sim, self.plane_params)

        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, self.env_lower, self.env_upper, self.num_per_row)
            self.envs.append(env)
            # add box
            self.box_pose.p.x = np.random.uniform(-0.1, 0.1)
            self.box_pose.p.y = np.random.uniform(-0.1, 0.1)
            self.box_pose.p.z = 0.5
            self.box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-0.2, 0.2))


            self.box_handle = self.gym.create_actor(env, l_boxes_asset[i], self.box_pose, "box", i, 0)
            color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
            self.gym.set_rigid_body_color(env, self.box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)


            # get global index of box in rigid body state tensor
            self.box_idx = self.gym.get_actor_rigid_body_index(env, self.box_handle, 0, gymapi.DOMAIN_SIM)
            self.box_idxs.append(self.box_idx)

            # add soto
            self.soto_handle = self.gym.create_actor(env, self.soto_asset, self.soto_pose, "soto", i, 1)
            self.l_handle.append(self.soto_handle)
            # set dof properties
            self.gym.set_actor_dof_properties(env, self.soto_handle, self.soto_dof_props)

            # set initial dof states
            self.gym.set_actor_dof_states(env, self.soto_handle, self.default_dof_state, gymapi.STATE_ALL)

            # set initial position targets
            self.gym.set_actor_dof_position_targets(env, self.soto_handle, self.default_dof_pos)

            # # get inital gripper pose
            # self.gripper_handle = self.gym.find_actor_rigid_body_handle(env, self.soto_handle, "gripper_base_link")
            #
            # self.gripper_pose = self.gym.get_rigid_transform(env, self.gripper_handle)
            #
            # self.init_pos_list.append([self.gripper_pose.p.x, self.gripper_pose.p.y, self.gripper_pose.p.z])
            # self.init_rot_list.append([self.gripper_pose.r.x, self.gripper_pose.r.y, self.gripper_pose.r.z, self.gripper_pose.r.w])

            # get global index of pieces in rigid body state tensor
            self.gripper_idx = self.gym.find_actor_rigid_body_index(env, self.soto_handle, "gripper_base_link", gymapi.DOMAIN_SIM)
            global_index = [self.gym.find_actor_rigid_body_index(env, self.soto_handle, i,gymapi.DOMAIN_SIM) for i in index_to_get]
            self.global_soto_indexs.append(global_index)

        # point camera at middle env
        self.cam_pos = gymapi.Vec3(4, 3, 2)
        self.cam_target = gymapi.Vec3(-4, -3, 0)
        self.middle_env = self.envs[self.num_envs // 2 + self.num_per_row // 2]
        self.gym.viewer_camera_look_at(self.viewer, self.middle_env, self.cam_pos, self.cam_target)
        x = 0
        print(self.l_handle)
        while not self.gym.query_viewer_has_closed(self.viewer):
            x+= 1
            # update viewer
            # set initial dof states
            k = np.abs(np.sin(x/100))
            self.soto_current = k*self.soto_upper_limits + (1-k)*self.soto_lower_limits
            self.default_dof_pos = self.soto_current

            # remember : important pieces to control are conveyor belt left base link/conveyor belt right base link
            self.default_dof_state["pos"] = self.default_dof_pos

            # send to torch
            self.default_dof_pos_tensor = to_torch(self.default_dof_pos, device=device)

            # set initial position targets
            for soto_handle in self.l_handle :
                # set dof states
                self.gym.set_actor_dof_states(env, soto_handle, self.default_dof_state, gymapi.STATE_ALL)

                # set position targets
                self.gym.set_actor_dof_position_targets(env, soto_handle, self.default_dof_pos)

            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

        # cleanup
        self.gym.destroy_viewer(self.viewer)

    def _init_observation_space(self):
        """
        Observation consists of :
                        dof_pos - 12
                        dof_vel - 12
                        roll - 1
                        pitch - 1
                        feet_contact_switches - 4
                        previous_actions -12
                        Mass - 1
                        COM_x - 1
                        COM_y - 1
                        Motor Strength - 12
                        Friction - 1
                        Local terrain height - 1
        :return: obs_space
        """
        limits_low = np.array(
            self.lower_bounds_joints +
            [0] * 12 +    # minimum values of joint velocities
            [-math.inf] +
            [-math.inf] +
            [0] * 4 +
            self.lower_bounds_joints +
            [0] +
            [-math.inf] +
            [-math.inf] +
            [motor_strength * self.cfg.domain_rand.motor_strength_range[0] for motor_strength in self.motor_strength] +
            [self.cfg.domain_rand.friction_range[0]] +
            [0]
        )
        limits_high = np.array(
            self.upper_bounds_joints +
            self.upper_bound_joint_velocities +
            [math.inf] +
            [math.inf] +
            [1] * 4 +
            self.upper_bounds_joints +
            [math.inf] +
            [math.inf] +
            [math.inf] +
            [motor_strength * self.cfg.domain_rand.motor_strength_range[1] for motor_strength in self.motor_strength] +
            [self.cfg.domain_rand.friction_range[1]] +
            [math.inf]
        )
        obs_space = gymspace.Box(limits_low, limits_high, dtype=np.float32)
        return obs_space

    def _init_action_space(self):
        """
        Upper and lower bounds of all 12 joints  of the robot extracted automatically from the URDF.
        Extraction function and values are defined in the accompanying scene class.
        Values can be found under <joint ...> <limit ... upper="" lower=""> </joint>

        :return: act_space -> gym.space.Box
        """
        lb = np.array(self.lower_bounds_joints)
        ub = np.array(self.upper_bounds_joints)
        act_space = gymspace.Box(lb, ub, dtype=np.float32)

        return act_space

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[:12] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[12:24] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[24:26] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[26:30] = 0.
        noise_vec[30:42] = 0.  # Previous action.
        noise_vec[42:45] = 0.  # Mass and COM
        noise_vec[45:57] = noise_scales.motor_strength * noise_level
        noise_vec[58] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements

        return noise_vec

    def compute_observations(self):
        """Overrides the base class observation computation to bring it in line with observations proposed in the RMA
        paper. The observation consists of two major parts - the environment variables and the state-action pair."""
        # self.compute_heading_deviation()
        feet_contact_switches = self._get_foot_status()
        local_terrain_height = self._get_local_terrain_height()

        self.X_t = torch.cat((self.dof_pos,
                              self.dof_vel,
                              self.base_rpy[0].unsqueeze(1),
                              self.base_rpy[1].unsqueeze(1),
                              feet_contact_switches
                              ), dim=-1)
        E_t = torch.cat((
            self.body_masses.unsqueeze(-1),
            self.body_com_x.unsqueeze(-1),
            self.body_com_y.unsqueeze(-1),
            self.torque_limits,
            self.friction_coeffs.squeeze(-1),
            local_terrain_height.unsqueeze(-1)
        ), dim=-1)

        self.obs_buf = torch.cat((self.X_t,
                                  self.last_actions,
                                  E_t), dim=-1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def compute_heading_deviation(self):
        commands_xy = self.commands[:, :2]
        command_normalized = torch.nan_to_num(
            (commands_xy.T / torch.norm(commands_xy, dim=1, p=2)).T
        )
        base_xy_velocity = self.base_lin_vel[:, :3]
        forward_orientation = torch.zeros_like(base_xy_velocity)
        forward_orientation[:, 0] = 1.0
        forward_orientation_quat = quat_rotate(self.base_quat, forward_orientation)
        forward_orientation_quat = normalize(forward_orientation_quat)[:, :2]

        self.heading_deviation = torch.acos((command_normalized * forward_orientation_quat).sum(dim=1))
        self.heading_deviation = (torch.sign(forward_orientation_quat[:, 1]) * self.heading_deviation).reshape((-1, 1))

        return self.heading_deviation

    # -------------- Reward functions begin below: --------------------------------#

    def _reward_forward(self):
        base_x_velocity = self.base_lin_vel[:, 0]
        MAX_FWD_VEL = 1.0
        forward = quat_apply(self.base_quat, self.forward_vec)
        # max_fwd_vel_tensor = torch.full_like(base_x_velocity, MAX_FWD_VEL)
        # reward = torch.abs(base_x_velocity - max_fwd_vel_tensor)
        diff = base_x_velocity - forward[:, 0]
        reward = torch.linalg.norm(diff.unsqueeze(-1), dim=1, ord=1)   # L1 norm

        return reward

    def _reward_maintain_forward(self):
        # Rewards forward motion at limited values (v_x)
        base_x_velocity = self.base_lin_vel[:, 0]
        MAX_FWD_VEL = 1.0
        max_fwd_vel_tensor = torch.full_like(base_x_velocity, MAX_FWD_VEL)
        reward = torch.fmin(base_x_velocity, max_fwd_vel_tensor)
        # print(" forward x vel", base_x_velocity)
        # print(" forward reward", reward)
        reward[reward < 0.0] = 0.0   # TODO: Check if this is right.
        # print(f"forward reward * {self.cfg.rewards.scales.forward}", reward)
        return reward

    def _reward_lateral_movement_rotation(self):
        # penalises lateral motion (v_y) and limiting angular velocity yaw
        reward = self.base_lin_vel[:, 1].pow(2).unsqueeze(-1).sum(dim=1) + self.base_ang_vel[:, 2].pow(2).unsqueeze(-1).sum(dim=1)   #TODO check with 1
        # print(f"lateral reward * {self.cfg.rewards.scales.lateral_movement_rotation}", reward)
        return reward

    def _reward_orientation(self):
        # orientation = self.projected_gravity[:, :2]
        orientation = torch.stack([self.base_rpy[0], self.base_rpy[1]], dim=1)
        reward = torch.sum(torch.square(orientation), dim=1)
        return reward

    def _reward_work(self):
        diff_joint_pos = self.actions - self.last_actions
        # torque_transpose = torch.transpose(self.torques, 0, 1)
        reward = torch.abs(torch.sum(torch.inner(self.torques, diff_joint_pos), dim=1))     # this is the L1 norm
        return reward

    def _reward_ground_impact(self):
        reward = torch.sum(torch.square(self.contact_forces[:, self.feet_indices, 2] -
                                        self.last_contact_forces[:, self.feet_indices, 2]), dim=1)  # only taking into account the vertical reaction, might need to check if parallel ground reaction makes sense.
        return reward

    def _reward_smoothness(self):
        reward = torch.sum(torch.square(self.torques-self.last_torques), dim=1)
        return reward

    def _reward_action_magnitude(self):
        reward = torch.sum(torch.square(self.actions), dim=1)
        return reward

    def _reward_joint_speed(self):
        reward = torch.sum(torch.square(self.last_dof_vel), dim=1)
        return reward

    def _reward_z_acceleration(self):
        reward = torch.square(self.base_lin_vel[:, 2])
        return reward

    def _reward_foot_slip(self):
        reward = 0  # TODO: Fix this
        return reward

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        reward = torch.square(self.base_lin_vel[:, 2])
        return reward

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_base_height(self):
        # Make sure the robot body maintains a minimum distance from the ground based on the z center of mass.
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.),
            dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        reward = torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
        return reward

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact,
                                dim=1)  # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > \
                         5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (
                    torch.norm(self.commands[:, :2], dim=1) < 0.1)

    # ---------------Reward functions end here ---------------------- #

    def _get_foot_status(self):
        # foot_status = torch.zeros_like(self.feet_indices)
        # feet_forces = self.contact_forces[:, self.feet_indices, :2]
        # for foot_index in self.feet_indices:
        #     contact = self.contact_forces[:, foot_index, 2]
        #     if contact > 0.:
        #         foot_status[foot_index] = 1.

        foot_status = torch.ones(self.cfg.env.num_envs, 4, device=self.device)
        feet_forces = self.contact_forces[:, self.feet_indices, 2]
        feet_switches = torch.where(feet_forces == 0., torch.tensor(0.0, device=self.device), foot_status)

        return feet_switches

    def _get_local_terrain_height(self):
        local_terrain_height = self.measured_heights
        # this gives the measured heights of all points below the body of the robot. Max of local terrain height is just
        # the max of each of the heights of the envs. This is in contrast to the paper which defines terrain height as
        # the max of the height below the robot feet.
        local_terrain_height = torch.max(local_terrain_height, dim=-1)[0]

        return local_terrain_height

    def _get_random_boxes(self,l_limit,w_limit,h_limit):
        length = random.uniform(l_limit[0],l_limit[1])
        width = random.uniform(w_limit[0],w_limit[1])
        height = random.uniform(h_limit[0],h_limit[1])
        return (length, width, height)
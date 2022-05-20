string joint_name          # equal to robot description joint name, also referenced in the motor configuration
float64 target_position    # target position in m or rad
bool absolute_position     # true if target_position is absolute, false if relative to current position
float64 velocity_limit     # [m/s] or [rad/s], negative values for no limitation
float64 end_velocity       # [m/s] or [rad/s]. Only used for new synchronous control
float64 acceleration_limit # [m/s^2] or [rad/s^2], negative values for no limitation
float64 jerk_limit         # [m/s^3] or [rad/s^3], negative values for no limitation
float64 effort_limit       # [mA] for now, [N] or [Nm] planned for later, negative values for no limitation
bool start_immediately     # true will drive to target_position immediately, false will start after current movement has reached its target
bool final_position        # should be true, false may be unsafe: does not stop at the target position.
Condition[] conditions     # Conditions to be verified before the movement begins
bool end_with_quickstop    # If set to True, the motor will perform a quickstop at the end of the movement and wait for SWICHED_ON_DISABLED

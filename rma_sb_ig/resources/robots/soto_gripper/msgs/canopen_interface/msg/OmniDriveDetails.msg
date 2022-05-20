# All units are SI units (N, Nm, W, rad, m/s)

# powered caster wheels
OmniDriveWheelDetails[] wheels

# Both wheels in combination cause the robot to
# - drive (in x and y directions),
# - rotate
# - slip
#
# Slipping can happen because the system is overdefined:
# Each wheel provides two degrees of freedom,
# but driving and rotation together only 'consume' three.
# Example: The PCWs point at each other and 'drive' at the same speed.
# The drive controller attempts to filter the motor commands
# such that slip is minimized, but due to dynamic effects this cannot
# be avoided fully.
#
# The values are referenced to the 'center' of the robot (the base link).
#
# The robot velocity and robot angular speed are available through /odom.

# How much force is excerted to move the robot linearily?
float32 driving_force_x
float32 driving_force_y

# How much mechanical power is excerted to move the robot linearily?
# (can be negative while braking)
float32 driving_power_x
float32 driving_power_y

# How much torque is excerted to turn the robot about its origin?
float32 rotation_torque
# How much mechanical power is excerted to turn the robot about its origin?
# (can be negative while braking)
float32 rotation_power

# Estimate of the average speed at which the wheels are slipping over the ground
# We cannot know this precisely, but we can assume that /odom will be wrong
# by about this much.
float32 slipping_speed
# Approximate mechanical power that is lost to slipping.
float32 slipping_power

# Indicator of the fraction of the current raw command that would
# go into slipping due to misalignment of the wheels.
# This is a unitless ratio.
float32 slipping_error_by_missteering

# Unit-less indicator of how much the commanded speeds were re-balanced
# between the wheels to reduce the above slipping error.
# If the speed commands were left exactly as-is, this is 0.
# If the speed commands are "changed completely", it is 1.
# For details, see https://magazino.atlassian.net/wiki/spaces/HD/pages/3163423229/Kinematics+calculations#2.B---Minimize-slip-due-to-unaligned-wheels-(geometry.minimize_slip)
float32 slipping_avoidance_rebalance

# Factor (between 0 and 1) by which the speed commands have been multiplied
# to fulfill constraints such as the safety fields.
float32 speed_scaledown
# What was the limiting factor for the wheel speeds? (human-readable)
string speed_scaledown_reason

# What is the maximum of the individual wheel's differences between
# commanded target steering angle and actual steering angle?
float32 wheels_steering_error

# Are all wheels ready to execute movement commands?
bool wheels_ready

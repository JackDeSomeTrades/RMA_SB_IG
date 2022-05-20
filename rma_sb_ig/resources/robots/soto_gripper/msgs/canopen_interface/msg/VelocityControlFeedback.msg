uint8 IDLE = 1
uint8 MOVING = 2
uint8 ERROR = 3

string[] joint_names            # array of names, equal to robot description joint name, also referenced in the motor configuration
float64[] current_velocities    # array of current velocities for each joint in m/s or rad/s
uint8[] states                  # array of states of the joint

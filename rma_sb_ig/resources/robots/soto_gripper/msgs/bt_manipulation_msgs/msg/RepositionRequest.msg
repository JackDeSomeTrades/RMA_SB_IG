# Constants defining the operations to be performed
uint8 NO_REPOSITION = 0                         # Move the robot base to target_pose
uint8 REPOSITION_OBJECT = 1                     # Reposition object 'obj_id' somewhere else
uint8 MOVE_TOWER = 2                            # Reposition with the robot
uint8 MOVE_TOWER_AND_BASE = 3                   # Reposition base and tower simultaneously
uint8 MOVE_ROBOT_BASE = 4                       # Move the robot base to target_pose
uint8 operation                                 # Operation to be performed in order to unblock the object to be grasped

geometry_msgs/TransformStamped target_transform # Target position for the robot
float32 reposition_distance                     # Reposition a given distance; use only target_transform if 0
moveit_msgs/CollisionObject[] objects_to_be_moved

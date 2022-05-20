 Conveyor commands
 ---
  /canopen_interface/canopen_conveyor/velocity_control : canopen_interface/VelocityControl

The message is magazino specific, it has been uploaded to git

string[] joint_names
float64[] target_velocities
float64[] target_accelerations

they should look like this:

joint_names: 
  - conveyor_belt_right
  - conveyor_belt_left
target_velocities: [0.0, 0.0]
target_accelerations: [0.0, 0.0]

You command the desired velocity, the motor will accelerate respecting the accelleration limit.
0.0 acceleration is unlimited. Stopping works with 0 velocity.

the maximal velocity is 0.247126, acceleration 0.6



Box pose 
---
/box_tracking/track_boxes/feedback: box_tracking_msgs/TrackBoxesActionFeedback 
The message is magazino specific, i will upload it


you are intersted in this part:
  items_in_manipulator: 
    - 
      header: 
        seq: 0
        stamp: 
          secs: 1644435600
          nsecs: 944192300
        frame_id: "gripper_surface_tip_link"
      pose: 
        position: 
          x: 0.0
          y: 0.0
          z: 0.0
        orientation: 
          x: 0.0
          y: 0.0
          z: 0.0
          w: 0.0
      id: "item"
      type: 
        key: ''
        db: "{\"barcode\": \"KLT-400_300_120\", \"name\": \"KLT-400_300_120\"}"

which is a list of moveit_msgs/CollisionObject. It will always be the first and only object.


Conveyor state
---

/joint_states_conveyor sensor_msgs/JointState
header: 
  seq: 265606
  stamp: 
    secs: 1644594598
    nsecs: 898155207
  frame_id: ''
name: 
  - conveyor_belt_left
  - conveyor_belt_right
position: [-496.10797397214407, -377.987103722273]
velocity: [0.0, 0.0]
effort: [-0.0522657485502864, 0.0391993114127148]
---


Joint states
---
for completness

/joint_states_canopen sensor_msgs/JointState

header: 
  seq: 266696
  stamp: 
    secs: 1644594625
    nsecs: 222765043
  frame_id: ''
name: 
  - backpack_tilt
  - gripper_base_x
  - gripper_rotate
  - gripper_y_left
  - gripper_y_right
  - press_jaw
  - vertical_axis
position: [-0.11501843486627447, 0.0019994343740495084, -1.559373280452999, 0.18277247277189754, 0.1812145267679092, 0.22765513918270905, 1.8295726463034192]
velocity: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
effort: [0.0, 0.2216068779145434, -0.10607994270744082, 0.0, 0.0, 0.3480983619635431, 0.0]

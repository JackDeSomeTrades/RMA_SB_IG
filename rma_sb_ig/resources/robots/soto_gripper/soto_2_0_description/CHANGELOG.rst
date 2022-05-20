^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package soto_2_0_description
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.26.0 (2020-10-02)
-------------------
* Update package xmls
* SW-38691 Update move tower config
  * Set reference to gripper_y_left_link
  * Add gripper y joint
* SW-38691 Disable move tower safety
* Allow gripper to rotate at a lower height
* Shift coordinates by 0.2885 m
  They were originally generated relative to the safety field origin
  instead of the base_link coordinate frame.

0.25.0 (2020-09-07)
-------------------

0.24.0 (2020-09-07)
-------------------

0.23.0 (2020-09-07)
-------------------
* Fix the final vertical axis movement during the homing sequence

0.22.0 (2020-09-04)
-------------------

0.21.0 (2020-09-03)
-------------------
* Add diagnostics for navigation cameras

0.20.0 (2020-09-03)
-------------------

0.19.0 (2020-09-03)
-------------------
* Add backpack homing to gripper
* Increase vertical axis final position
* SW-38379 Fix backpack_tilt movement direction
* SW-38370 Tilt back backpack before gripper_rotate and vertical axis homing
  In order not to crash the gripper cover with the backpack boxes

0.18.0 (2020-09-02)
-------------------
* Block also gripper_rotate when backpack tilted towards gripper
* 0.17.0
* SW-37584 Prevent crash between gripper covers and backpack boxes

0.17.0 (2020-08-31)
-------------------
* Fix design elements and wheel inertias
  Inflated wheel inertias and added dynamics to control the drifting in the joints
* Rename minus and plus with min and max
* Remove force condition TODO
  It seems like force_condition means that a joint limit can be specified
  in runtime through the /joint_conditions topic (deduced from
  https://bitbucket.org/Magazino/book_gripper_safety_filter/commits/759ae18ec96170c5c78d9ad8c0c77978a4b46e03)
  This makes sense for Toru, since it can't telescope if the ceiling
  isn't high enough, but not for Soto.
* Use calib pattern height also fro front party height
* Add more anchors to safety rules
* Use more name constants, fix extended_gripper_base_collision_with_front_party_vertical
* Create some constants, tune values
* Add backpack_tilt to homing sequence
* Avoid crashing with calibration pattern under backpack
* Split rotation_in_vert_min_pos in 2
* Translate gripper_rotate values so 0 means looking to backpack
* Prevent crashing with side lower covers
* Organize rules
* Add rotation_in_vert_min_pos rule
* Remove vertical axis from homing required list
* Move vertical axis up after homing
* Add backpack, drive to rotatable pose before rotating gripper
* SW-37740 Add power supply lines config
* Change to gripper_foremost_center_link to gripper_surface_tip_link

0.16.0 (2020-08-13)
-------------------
* Limit vertical axis speed in the homing sequence

0.15.0 (2020-08-03)
-------------------
* Adapt retract_gripper_sequence
* First home gripper_rotate, then move vert axis to min
* Home gripper_base_x and then gripper_y's
* Add gripper_base_x to the homing blacklist
* Add gripper_rotate presence if standalone
* Fix homing sequence
* Fix safety filter

0.14.0 (2020-06-24)
-------------------

0.13.0 (2020-06-23)
-------------------
* Add safety and update movement config
* Add design elements

0.12.0 (2020-06-22)
-------------------
* Add movement and empty power supply lines configs

0.11.0 (2020-06-15)
-------------------

0.10.0 (2020-05-20)
-------------------

0.9.0 (2020-04-07)
------------------
* Add support for existing xacro naming convention

0.8.0 (2020-03-25)
------------------
* Sort the footprint points CCW

0.7.0 (2020-03-11)
------------------
* Shift klt_press include to soto to mirror component structure

0.6.0 (2020-03-10)
------------------
* Add updated SOTO footprint based on laser field

0.5.0 (2020-03-05)
------------------

0.4.0 (2020-02-05)
------------------
* Activate backpack_soto_4_0 in top level description file

0.3.0 (2020-01-21)
------------------

0.2.0 (2020-01-15)
------------------
* Add conveyor_gripper_1_0_description
* Add support for vertical_axis_soto_2_0_description
* Add top level soto_2_0_description package

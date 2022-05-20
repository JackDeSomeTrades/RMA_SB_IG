^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package conveyor_gripper_1_0_description
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.26.0 (2020-10-02)
-------------------
* Update package xmls
* SW-38971 Change conveyors direction
* Add barcode_search_frame
* Add ability to configure conveyor_belt_left_barcode_camera joint
* Install old 64 gripper rotate gear

0.25.0 (2020-09-07)
-------------------
* Fix the safety rules for conveyor gripper
* Update gripper_rotation gear from WPLFE064-64 to WPLFE064-100

0.24.0 (2020-09-07)
-------------------
* Split the conveyor_gripper_1_0 safety rules into multiple files

0.23.0 (2020-09-07)
-------------------

0.22.0 (2020-09-04)
-------------------

0.21.0 (2020-09-03)
-------------------

0.20.0 (2020-09-03)
-------------------

0.19.0 (2020-09-03)
-------------------
* Add conveyor_belt_width parameter for manipulation use
* SW-38370 Move all rules and anchors to one file
  So that the anchors are not considered safety rules

0.18.0 (2020-09-02)
-------------------
* 0.17.0
* Reduce the gripper homing ovreshoots

0.17.0 (2020-08-31)
-------------------
* Rotate gripper_rotate frame by 180 degrees
* Change gripper_rotate limits so that 0 means looking to backpack
* Reduce gripper_rotate homing speed
  Changed from 1.51353 and 0.151353 (reverse) to 0.348449 and 0.0348449
* Change to gripper_foremost_center_link to gripper_surface_tip_link

0.16.0 (2020-08-13)
-------------------
* Update gripper_rotate inertia when covers are made of metal
* SW-32995 Fix diagnostics
* Add jerk to gripper_rotate

0.15.0 (2020-08-03)
-------------------
* Allow moving to homing position
* Include limit in gripper_lateral_frame.xacro
* Fix griper_rotate homing switch distance
* Restore includes and properties in gripper_base.xacro
* Fix duplicated gripper_rotate_link
* Remove standalone condition for gripper_rotate
* Move gripper_rotate to conveyor_gripper_1_0
* Add gripper_rotate limit switch
* Fix gripper_rotate homing direction
* Relax backpack constraints
* Fix homing direction for gripper_base_x
* Fix diagnostics
* Fix meta configs
* Adapted to conveyor_y rack renaming
* Addressed comments
* Updated with values measured from real robot
* Add gripper_rotate presence if standalone
* Fix homing sequence
* Fix safety filter
* Final update before release
* Fix the safety rules for conveyor_gripper_1_0

0.14.0 (2020-06-24)
-------------------

0.13.0 (2020-06-23)
-------------------
* Add safety filter rules to conveyor_gripper_1_0
* Update conveyor_gripper_1_0_description for Soto 2.0

0.12.0 (2020-06-22)
-------------------
* SW-35677: Add missing diagnostics

0.11.0 (2020-06-15)
-------------------

0.10.0 (2020-05-20)
-------------------
* SW-34379 revert the renaming of the urdf yamls

0.9.0 (2020-04-07)
------------------
* copy component configurations from dimensioning_notebook

0.8.0 (2020-03-25)
------------------

0.7.0 (2020-03-11)
------------------

0.6.0 (2020-03-10)
------------------
* Add depth camera to conveyor_gripper

0.5.0 (2020-03-05)
------------------

0.4.0 (2020-02-05)
------------------
* Add prismatic joints for conveyor_belt_left and conveyor_belt_right
* Add cman configurable limits for conveyor_gripper and vertical_axis
* Fix gripper inertials and remove gripper sensor collisions

0.3.0 (2020-01-21)
------------------
* Move standalone argument to top level conveyor xacro
* Update links names
* Add missing configurations

0.2.0 (2020-01-15)
------------------
* Add conveyor_gripper_1_0_description

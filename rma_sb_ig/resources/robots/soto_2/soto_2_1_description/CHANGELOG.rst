^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package soto_2_1_description
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.9.2 (2022-04-07)
------------------
* Remove backpack movement from retract gripper sequence

0.9.1 (2022-04-07)
------------------
* Simplify meshes

0.9.0 (2022-03-29)
------------------
* Prevent gripper_y_left movement into vertical axis energy chain.
  It shouldn't move lower than gripper_y_left_with_backpack_pillar
  no matter what the gripper_base_x position is.
  Also add missing
  gripper_base_extension_gripper_y_right_with_rear_window_left_pillar
  rule.
  Adapt homing sequence to new gripper_ys_with_rear_window_pillars rule
  Increase rotation trigger range for collisions in rear window
  Closes: SW-55376

0.8.0 (2022-03-08)
------------------
* Add another range in the front window to allow further extension
  Closes: SOTO-705

0.7.0 (2022-02-09)
------------------
* Increase trigger range for gripper_rear_cover_with_backpack_lower_cover_vertical
  Apply bigger rotation range also to other crashes when vertical axis drives downwards
  Closes: SW-53764

0.6.0 (2022-02-03)
------------------
* Fix extension into vertical axis energy chain
  Closes: SW-54252

0.5.0 (2021-12-22)
------------------
* Set vertical frontal limit config
  Relates: SW-53765
* Prevent collision between gripper rear cover and backpack lower cover
  Closes: SW-53764
* Prevent rotation from one sector to another with low vert axis
  Closes: SW-53454
* Remove obsolete gripper cover with backpack rules
  Closes: SW-53584
* Update retract gripper sequence
  Relates: SW-52008

0.4.0 (2021-12-14)
------------------
* Add config for constrained rotation
  Relates: SW-52909

0.3.0 (2021-11-05)
------------------
* Set move tower reference frame
  Relates: SW-52077
* Allow moving gripper_ys lower when extended through window
  Prevent collision between gripper_ys and vertical axis energy chain
  Closes: SW-52121

0.2.1 (2021-11-03)
------------------
* Remove diagnostics
  Relates: SW-52077

0.2.0 (2021-11-02)
------------------
* Adapt safety filter rules to Soto 2.1
  Closes: SW-47670
* Update rules as in Soto 2.0
* Update retract gripper sequence
  Relates: SW-51967
* Configure homing sequence
  Closes: SW-51967
* Fix naming from soto_backpack_5_0 to soto_backpack_5_1
  The repositories soto_backpack_5_0 and soto_backpack_5_0_description
  were wrongly named. To fix this they have been copied to new
  repositories
  Fixes: SW-51161
* Configure move tower for targeted rotations
  Relates: SW-48102

0.1.0 (2021-08-26)
------------------
* Add new design elements
  Relates: SW-48091
* Merged in benni/add_soto_2_1_description (pull request #1)
  Add soto 2.1 description files
  Approved-by: Nick Lamprianidis
* Initial commit

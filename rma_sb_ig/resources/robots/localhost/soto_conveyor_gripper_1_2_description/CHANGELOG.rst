^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package soto_conveyor_gripper_1_2_description
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.20.0 (2022-05-17)
-------------------
* Update conveyor distance sensor poses
  - Lower position by 1.75cm
  - Update rotations
  - Add cutout in gripper_y meshes
  - Y-mirror gripper_y_right meshes

0.19.0 (2022-05-10)
-------------------
* Add press_calibration_link

0.18.0 (2022-05-05)
-------------------
* Add movement config
  Relates to: SW-57280

0.17.1 (2022-04-26)
-------------------
* Refine gripper base link collision model
  Relates: SW-57241

0.17.0 (2022-03-29)
-------------------
* Adapt gripper safety range sensor name to Soto 2.1 naming
  Relates to: SW-55772

0.16.0 (2022-03-10)
-------------------
* Simplify gripper base and gripper base x meshes
* Simplify distance sensor mesh
* Remove camera and safety sensors mount from gripper base mesh
  Relates: SW-56013
* Further simplify barcode camera mesh
* Simplify reflective sensor mesh
* Simplify barcode camera mesh
* Remove lens cap from barcode camera mesh
* Update camera poses
  Relates: SW-56013

0.15.0 (2022-03-08)
-------------------
* Elevate safety sensors 15 mm

0.14.0 (2022-03-08)
-------------------
* Change gripper_base_x limits
  Closes: SOTO-704

0.13.0 (2022-02-23)
-------------------
* Merged in kai/SW-55396/descrease-nominal-torque (pull request #27)
  Decreases nominal torque of gripper_rotate
  * Decreases nominal torque of gripper_rotate
  Adds EngineeringLimit which limits the motor's nominal current to 6A to
  prevent it from dying when the nominal torque is applied for a very long
  time. Also increases peak current duration to make sure i2t will no
  Approved-by: Daniel Zimmermann
  Approved-by: Alejandro Yunta
  Approved-by: Benjamin Schimke
* Increased belt pretension
  The belt pretension can be increased after installing an additional
  bearing on the opposite side of the conveyor motor
  Solves: HW-707

0.12.1 (2022-02-03)
-------------------
* Update conveyor gripper board diagnostic name
  Relates: SW-54935

0.12.0 (2022-01-24)
-------------------
* Limit gripper_rotate peak current

0.11.0 (2021-12-22)
-------------------
* Update safety sensor poses
  Relates: HW-663

0.10.0 (2021-12-16)
-------------------
* Adapt sum of gripper_ys to be equal to sum of URDF min + max
  Closes: SW-53622
* Lift conveyor distance sensors

0.9.2 (2021-12-13)
------------------
* Change conveyor_gripper references to gripper

0.9.1 (2021-12-07)
------------------
* Fix gripper_surface_tip_link position
  Closes: SW-53475

0.9.0 (2021-12-01)
------------------
* Elongate conveyors
  Closes: SW-51863

0.8.0 (2021-11-22)
------------------
* Increased belt pretension because of error in datasheet

0.7.0 (2021-11-08)
------------------
* Remove manipulator_rotation diagnostics
* Add operational gripper diagnostics
* Fix manipulation sensor diagnostics
* Fix camera diagnostics
  Relates: SW-52077
* Reduce conveyor belt max accelerations
  Closes: SW-52188

0.6.0 (2021-11-05)
------------------
* Update distance sensor names
* Fix sensor/manipulation diagnostics
  Relates: SW-52077
* Fix gripper_y joint states
* Adjusts belt pretension for all belt drive trains
  Increases the belt pretension to the radial force limit of the
  preceeding gear or even exceeds this limit to not reduce the torque of
  belt drives in the field.
  More information can be found here
  https://magazino.atlassian.net/wiki/spaces/HD/pages/3371040945/How+to+fix+the+incorrect+belt+pretension#Detailed-data-of-drive-train%E2%80%99s-transmissible-torque-for-all-robots
  Relates to: SW-51460

0.5.0 (2021-11-02)
------------------
* Rename base_min to gripper_base_min to be used in the homing sequence
* Adapt safety filter rules to Soto 2.1
  Relates to: SW-47670

0.4.0 (2021-11-02)
------------------
* Sync barcode camera frame name with the camera
* Change conveyor joints to prismatic
  Relates: SW-47600
* Set joint dynamics
  Relates: SW-46679

0.3.0 (2021-10-21)
------------------
* Fix gripper depth camera orientation
* Reduce URDF limits
* Merged in Inka/SW-48164/camera-diagnostics (pull request #3)
  update camera diagnostics
* Fix cables collision
* Fix limits sign
* Fix broken build by removing spaces
* Extract CAD parameters
  Closes: SW-51149
* Fix gripper_rotate homing direction
  Fixes: SW-51139
* Diameter of conveyor pulley was increased

0.2.0 (2021-09-16)
------------------
* Add conveyor belt length and radius
  Relates: SW-46679
* Add inertia to surface foremost links
  Relates: SW-46679

0.1.0 (2021-08-26)
------------------
* Add new urdf
  Relates: SW-48091
* Rename package
  Relates: SW-48091
* Move resources
* Merged in benni/Fix-diagnostics-update (pull request #179)
  Update regex
  Approved-by: Nick Lamprianidis
* Merged in benni/SW-48356-Cleanup-fix-hw-diagnostics (pull request #178)
  Benni/SW-48356 Cleanup fix hw diagnostics
  Approved-by: Alejandro Yunta
* Prevent gripper crashing with pillars when extending through curtains
  Prevent gripper_ys crashing with pillars when extending / extended base
  Add missing front low rotation rule
  Add missing corners for gripper_base_extension
  Increase URDF limits to allow docking misalignment compensation
  Closes: SW-46564
* Merged in SW-44486-update-parameters-for-polarity-fix (pull request #153)
  SW-44486 update parameters for polarity fix
  Approved-by: Michael Enlin
  Approved-by: Alejandro Yunta
  Approved-by: Kai Franke
* Load cman joints also for belt_right
* Add conveyors to URDF configuration yaml
  Use autogenerated velocity and effort in Conv. Gr. 1.0 and 1.2
  Closes: SW-45610
* Set homing switch in C.G. 1.2 gripper_rotate to 2
* Move comments
* Update conveyor gripper rotate motor parts
  Reduce max_accelerations for smoother movements
* Merged in SW-39253-Setup-conveyor-gripper-rotate-motor (pull request #145)
  SW-39253 Update conveyor gripper rotate motor parts
  * Update conveyor gripper rotate motor parts
  * Reduce max_acceleration for smoother movement
  Approved-by: Michael Enlin
  Approved-by: Kai Franke
  Approved-by: Alejandro Yunta
* always use 25 teeths pinion
* SW-42992: better estimate for barcode_search_pose
* Fix conveyor gripper 1.2 version
* Fix gripper depth camera in conveyor gripper 1.2
* fixed formatting
* fixed CHANGELOG
* adressed PR comments
* added data from CAD from Oli
* added jerk
* added conveyor_gripper_1_2

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package vertical_axis_soto_2_0_description
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.26.0 (2020-10-02)
-------------------
* Update package xmls

0.25.0 (2020-09-07)
-------------------

0.24.0 (2020-09-07)
-------------------

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

0.18.0 (2020-09-02)
-------------------
* 0.17.0

0.17.0 (2020-08-31)
-------------------

0.16.0 (2020-08-13)
-------------------
* SW-32995 Fix diagnostics

0.15.0 (2020-08-03)
-------------------
* Decrease vertical axis minimum urdf limit for now
* Increase vert axis urdf min for now
* Remove gripper_rotate from vertical axis description
* Move gripper_rotate to conveyor_gripper_1_0
* Fix vertical axis configuration
* Fix diagnostics
* Fix meta configs
* SW-36488 Fix vertical axis collision origin
* Add vertical axis stopping distance
* Added missing data from vertical axis
* Final update for vertical axis before release

0.14.0 (2020-06-24)
-------------------

0.13.0 (2020-06-23)
-------------------
* Add exported SOTO 2.0 vertical_axis

0.12.0 (2020-06-22)
-------------------
* SW-35677: Add missing diagnostics

0.11.0 (2020-06-15)
-------------------
* Add the three active PCBs

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
* Shift klt_press include to soto to mirror component structure
* SW-32395 Add klt press diagnostics and complete vertical axis diagnostics

0.6.0 (2020-03-10)
------------------
* Update limits from dimensioning notebook
* Update light curtain heights relative to manipulator_base_link

0.5.0 (2020-03-05)
------------------
* SW-32099 Prepare klt press for simulation
  * Optimize link inertias for vertical axis and press. Multiplying factors are added to help stabilize the simulated model. The "normal" inertia components are kept for future reference, in case a more elegant solution is found
  * Move inertia and collision model from right jaw to primary jaw link. As the driving jaw joint is connected to press_jaw_link, this link needs to interact with an object for the mimic joints the respect the imposed constraints
  * Remove light barrier collision models
* Add klt_press_1_0 to vertical_axis_soto_2_0

0.4.0 (2020-02-05)
------------------
* Add cman configurable limits for conveyor_gripper and vertical_axis

0.3.0 (2020-01-21)
------------------
* Add standalone argument to vertical_axis
* Add missing configurations

0.2.0 (2020-01-15)
------------------
* Add vertical_axis_soto_2_0_description package

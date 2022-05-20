^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package klt_press_1_0_description
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
* Increase klt_press_homing speed

0.21.0 (2020-09-03)
-------------------

0.20.0 (2020-09-03)
-------------------
* SW-38245: Reduce the acceleration so the motor does not throw overcurrents

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
* Match press_jaw_right_link and press_jaw_link inertials
  Used orignal intertial which is wider. Point inertial was making press_jaw_right_link unstable around the limits

0.15.0 (2020-08-03)
-------------------

0.14.0 (2020-06-24)
-------------------
* SW-35807: Fix diagnostics configuration and gear used

0.13.0 (2020-06-23)
-------------------
* Update klt_press transformation

0.12.0 (2020-06-22)
-------------------

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
* SW-32395 Add klt press diagnostics and complete vertical axis diagnostics

0.6.0 (2020-03-10)
------------------
* Update limits from dimensioning notebook
* Add downward facing camera to KLT press

0.5.0 (2020-03-05)
------------------
* SW-32099 Prepare klt press for simulation
  * Optimize link inertias for vertical axis and press. Multiplying factors are added to help stabilize the simulated model. The "normal" inertia components are kept for future reference, in case a more elegant solution is found
  * Move inertia and collision model from right jaw to primary jaw link. As the driving jaw joint is connected to press_jaw_link, this link needs to interact with an object for the mimic joints the respect the imposed constraints
  * Remove light barrier collision models
* Convert light barriers to point_inertials (again)
* Fix press_jaw joint transformation
* Set standalone back to false
* Fix joint limit parameters
* Fix naming and add missing upper_limit_link
* Convert light barriers to point inertials
* Add klt_press_1_0 to vertical_axis_soto_2_0
* Add mimic prismatic joints
* Add klt_press_1_0_description

0.4.0 (2020-02-05)
------------------

0.3.0 (2020-01-21)
------------------

0.2.0 (2020-01-15)
------------------

0.1.0 (2020-01-07)
------------------

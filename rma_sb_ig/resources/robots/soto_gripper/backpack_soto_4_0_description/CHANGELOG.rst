^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package backpack_soto_4_0_description
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.26.0 (2020-10-02)
-------------------
* SW-39134 Use db frame convention for backpack layers
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
* Decrease backpack_tilt max jerk and max speed
  Decrease max jerk from inf to 10
  Decrease max vel from 0.4799 to 0.24

0.20.0 (2020-09-03)
-------------------

0.19.0 (2020-09-03)
-------------------
* Change inverted polarity
* SW-38379 Fix backpack_tilt movement direction
* SW-38099 Fix backpack_soto_4_0 URDF limits

0.18.0 (2020-09-02)
-------------------
* 0.17.0
* Add frame offset and board height to backpack config

0.17.0 (2020-08-31)
-------------------

0.16.0 (2020-08-13)
-------------------
* SW-32995 Fix diagnostics
* Autogenerate backpack_soto_4_0_urdf.yaml
* Simplify dependencies on backpack_tilt
* Reconfigure backpack_tilt joint
  Fix the double movements of the backpack layers (translation along pivot bar and rotation along backpack layer tilt)
* Add backpack homing switch
* Use more realistic tolerances
* Fix backpack limits
* Fix speed and torque
* Add fake gear with ratio from cam to layer angle
  Update effort and velocity values
* Include backpack_tilt in URDF

0.15.0 (2020-08-03)
-------------------
* Remove not existing belt from backpack
* Increase belt pretension to gearbox max axial load
* Fix backpack tilt gear
* - Relax backpack constraints
  - Fix homing direction for gripper_base_x
* Replace backpack Soto 4.0 motor with PD6

0.14.0 (2020-06-24)
-------------------

0.13.0 (2020-06-23)
-------------------
* Add nav_3d_cam_rear and update charger_connector
* Update meshes and transformations for backpack links

0.12.0 (2020-06-22)
-------------------

0.11.0 (2020-06-15)
-------------------

0.10.0 (2020-05-20)
-------------------

0.9.0 (2020-04-07)
------------------
* Add pseudo-configurable file for layer_configuration
* Improve backpack model 
* Copy component configurations from dimensioning_notebook

0.8.0 (2020-03-25)
------------------

0.7.0 (2020-03-11)
------------------

0.6.0 (2020-03-10)
------------------

0.5.0 (2020-03-05)
------------------
* SW-31938 Update backpack
  * Remove backpack layer link collision geometry (a simplified model is defined in backpack_soto_4_0_description_sim)
  * Make backpack joint configurable

0.4.0 (2020-02-05)
------------------
* Initial commit for backpack_soto_4_0_description

0.3.0 (2020-01-21)
------------------
* Match backpack_soto_4_0_description version consistent with others
* Change from backpack_rotate to backpack_tilt
* Add missing configurations

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package soto_drive_base_2_1_description
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.6.0 (2022-03-10)
------------------
* Simplify meshes

0.5.0 (2021-12-12)
------------------
* Lift caster links
  Relates: SW-48148
* Update pcw links
  - Remove meshes
  - Fix inertials
  Relates: SW-48148

0.4.2 (2021-12-02)
------------------
* Improve base_link collision model

0.4.1 (2021-11-30)
------------------
* SW-49290 Publish diagnostics in argo_omni_drive

0.4.0 (2021-11-27)
------------------
* Update drive base model
  - Rotate rear wheel to be oriented forward by default
  - Use a cylinder instead of the mesh file for the wheels
  - Fix caster positions
  - Add wheel dynamics
  Relates: SW-48064

0.3.0 (2021-11-01)
------------------
* Add imu_link
  Moved from backpack
  Closes: SW-51999

0.2.0 (2021-09-16)
------------------
* Rename laser scanners
* Fix caster positions

0.1.0 (2021-08-26)
------------------
* Update colors
* Add urdf
  Relates: SW-48091
* Initial commit

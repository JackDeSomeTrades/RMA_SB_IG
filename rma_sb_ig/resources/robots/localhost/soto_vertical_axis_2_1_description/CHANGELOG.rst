^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package soto_vertical_axis_2_1_description
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.10.0 (2022-05-17)
-------------------
* Fix potential collision between screw and paddles
  Reduces vertical axis max position to prevent collision between the
  screws of the slightly rotated conveyor distance sensor and the KLT
  paddles.
  Relates to: HW-712

0.9.0 (2022-05-17)
------------------
* Revert "Adapt urdf_max to new conveyor sensors position"
  This reverts commit 444b04f41622de9382417b10ea324148b3cab853.

0.8.0 (2022-03-10)
------------------
* Simplify meshes
* Fixes home offset for vertical axis
  Updates limit switch values with values from CAD
  Solves: HW-86

0.7.0 (2022-03-01)
------------------
* SW-55652 Decrease vertical axis speed limit

0.6.0 (2021-12-17)
------------------
* Adapt urdf_max to new conveyor sensors position
  Closes: SW-53661
* Use 0.45 m/s as max velocity

0.5.1 (2021-12-09)
------------------
* Fix vertical_axis joint origin
  Closes: SW-53543

0.5.0 (2021-11-08)
------------------
* Update diagnostics
  Relates: SW-52077
* Fix camera diagnostics
  Relates: SW-52077
* Adjust belt pretension to corrected value \n Relates to: SW-51460

0.4.0 (2021-11-03)
------------------
* Fix nav cam namespace in diagnostics
* Add blue spot link

0.3.0 (2021-11-02)
------------------
* Set joint dynamics
  Relates: SW-46679

0.2.0 (2021-10-21)
------------------
* Add 'overdrive' to overdriven motor names
* Reduce the minimum, since it still has some buffer distance in Soto 9
* Increased vertical axis belt pretention
  The pretention most likely does not need to be that high since there are
  two of the belts transmitting power. The increase is needed to reach the
  expected peak torque
* changed from VA 2.0 to VA 2.1 values
* Fix home offset
* Set neutral position as urdf_min
* Update dimensions from CAD
  Closes: SW-51148
* Merged in Inka/SW-48164/camera-diagnostics (pull request #2)
  update camera diagnostics

0.1.1 (2021-09-16)
------------------
* Fix camera position

0.1.0 (2021-08-26)
------------------
* Add urdf
  Relates: SW-48091
* Initial commit

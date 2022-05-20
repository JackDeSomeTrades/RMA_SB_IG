^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package bt_manipulation_msgs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.7.0 (2021-10-21)
------------------
* Copy HomeJoints and MoveToTargetTransform actions for bt_grasping_msgs
* SW-39202 Add RewindTower action
* Add GetAlignedTransform action
  Relates: SW-48102
* Update package

1.6.0 (2021-05-12)
------------------
* Extend Interface by col_obj, add Press Closes: SW-46083

1.5.0 (2021-03-17)
------------------
* SW-43155 Clarify tower safety state message

1.4.0 (2020-08-06)
------------------
* SW-32862 Add AtHomeState msg

1.3.0 (2020-04-30)
------------------
* add ERROR_CONFLICTING_MEASUREMENT (SW-33993)

1.2.0 (2019-12-19)
------------------
* Add current_reachable_height, comments
* SW-29067 Add shape feedback to execute manipualtion
* Clarify and rename allowed height
* Clarify remaining_distance is positive or zero
* Add TowerSafetyState message

1.1.0 (2019-06-13)
------------------
* add ERROR_MEASUREMENT_FAILED

1.0.0 (2019-05-03)
------------------
* Add error codes for wrong item
* Add a col env to the Result of plan and execution
* SW-14723 Add measure flag to plan-manipualtion-goal
* Use float32 and comment
* added reposition distance

* Add error codes for wrong item
* Merged in maerz/add-col-env-to-result (pull request #6)
  Add a col env to the Result of plan and execution
  Approved-by: Steffen Rhl <ruehl@magazino.eu>
  Approved-by: Dominik Meinzer <meinzer@magazino.eu>
  Approved-by: Guglielmo Gemignani <gemignani@magazino.eu>
* Add a col env to the Result of plan and execution
* Merged in maerz/SW-14723-add-measure-flag (pull request #5)
  SW-14723 Add measure flag to plan-manipualtion-goal
  Approved-by: Guglielmo Gemignani <gemignani@magazino.eu>
  Approved-by: Dominik Meinzer <meinzer@magazino.eu>
  Approved-by: Calvin Ngan <ngan@magazino.eu>
  Approved-by: Steffen Rhl <ruehl@magazino.eu>
* SW-14723 Add measure flag to plan-manipualtion-goal
* Merged in unstable/reposition-distance (pull request #3)
  Added reposition distance to reposition msg
  Approved-by: Guglielmo Gemignani <gemignani@magazino.eu>
  Approved-by: Jorge Santos Simn <santos@magazino.eu>
  Approved-by: Steffen Rhl <ruehl@magazino.eu>
* Use float32 and comment
* added reposition distance
* Contributors: Jorge Santos Simn, Marco Bassa, Maxi Mrz, Michael Maerz, Steffen Rhl

* Add error codes for wrong item
* Merged in maerz/add-col-env-to-result (pull request #6)
  Add a col env to the Result of plan and execution
  Approved-by: Steffen Rhl <ruehl@magazino.eu>
  Approved-by: Dominik Meinzer <meinzer@magazino.eu>
  Approved-by: Guglielmo Gemignani <gemignani@magazino.eu>
* Add a col env to the Result of plan and execution
* Merged in maerz/SW-14723-add-measure-flag (pull request #5)
  SW-14723 Add measure flag to plan-manipualtion-goal
  Approved-by: Guglielmo Gemignani <gemignani@magazino.eu>
  Approved-by: Dominik Meinzer <meinzer@magazino.eu>
  Approved-by: Calvin Ngan <ngan@magazino.eu>
  Approved-by: Steffen Rhl <ruehl@magazino.eu>
* SW-14723 Add measure flag to plan-manipualtion-goal
* Merged in unstable/reposition-distance (pull request #3)
  Added reposition distance to reposition msg
  Approved-by: Guglielmo Gemignani <gemignani@magazino.eu>
  Approved-by: Jorge Santos Simn <santos@magazino.eu>
  Approved-by: Steffen Rhl <ruehl@magazino.eu>
* Use float32 and comment
* added reposition distance
* Contributors: Jorge Santos Simn, Marco Bassa, Maxi Mrz, Michael Maerz, Steffen Rhl

0.0.2 (2019-01-17)
------------------
* Initial commit

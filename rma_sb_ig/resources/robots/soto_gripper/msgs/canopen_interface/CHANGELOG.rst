^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package canopen_interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.19.0 (2021-11-10)
-------------------
* Update maintainer

0.18.0 (2021-08-12)
-------------------
* Improve documentation
* Remove wheels_steering_status from OmniDriveDetails
* Add slipping error correction info
* Add motor powers to omni drive wheel details message
* Update for rename orientation -> steering
* Add extra fields and create a sub-message for wheels
* Add additional fields to OmniDrive details message

0.17.0 (2021-01-27)
-------------------
* Add OmniDriveDetails
* Update maintainer

0.16.0 (2020-10-16)
-------------------
* SW-39820: Add more detailed initialization data
* Add feedback for velocity control

0.15.0 (2020-08-03)
-------------------
* SW-36404: Add max acc to VelocityCommand

0.14.0 (2020-06-19)
-------------------
* SW-33183: Add velocity control message

0.13.0 (2020-02-12)
-------------------
* Add exit code if another action is alredy running

0.12.0 (2019-11-20)
-------------------
* Add impulse to action results

0.11.0 (2019-09-02)
-------------------
* Add cfgword amd opmode actions to configure IO-Link sensors

0.10.0 (2019-06-03)
-------------------
* Add DiffDriveEfforts message type

0.9.0 (2019-04-10)
------------------
* Refactor safety-limit reporting
  Add new message entry 'max_individual_wheel_speed'

0.8.5 (2018-12-07)
------------------
* Added device reset service message.

0.8.4 (2018-11-20)
------------------
* diff drive profile position: Added action file

0.8.3 (2018-10-01)
------------------
* feat(HominAction): deal with vector of joints

0.8.2 (2018-09-13)
------------------
* Change torque type to double

0.8.1 (2018-08-21)
------------------
* changed result to vector
* Feedback as a vector
* Add energy feedback in SPP

0.8.0 (2018-08-13)
------------------
* fixes typo in enumeration element name
* Include support to expose energy consumption after the action

0.7.1 (2018-08-06)
------------------
* feat(HomeAction): add filtered feedback

0.7.0 (2018-07-24)
------------------
* feat(homing): add action

0.6.0 (2018-06-15)
------------------
* Bool to UInt8

0.5.10 (2018-06-06)
-------------------
* Remove Kai from the maintainers lisr
* Added base type in the message

0.5.9 (2018-04-23)
------------------
* add new DiffDriveSafetyLimits message

0.5.8 (2018-04-17)
------------------
* add Jose Pardeiro as package maintainer
* Added the PWMmessage

0.5.7 (2017-10-09)
------------------
* added header to the controller debug message

0.5.6 (2017-08-23)
------------------
* SW-5585 adding async_target flag to SPP actions

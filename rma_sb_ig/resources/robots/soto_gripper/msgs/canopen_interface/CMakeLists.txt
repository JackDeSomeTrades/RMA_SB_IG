cmake_minimum_required(VERSION 3.0.2)
project(canopen_interface)

find_package(catkin REQUIRED COMPONENTS
  actionlib_msgs
  message_generation
  dynamic_reconfigure
  sensor_msgs
  std_msgs
)

add_message_files(
  FILES
    DeviceState.msg
    DiffDriveEfforts.msg
    DiffDriveSafetyLimits.msg
    IOMessage.msg
    IOMessageTrigger.msg
    JointStatesTrigger.msg
    OmniDriveDetails.msg
    OmniDriveWheelDetails.msg
    OperationMode.msg
    ProfilePosition.msg
    ProfileTorque.msg
    ProfileVelocity.msg
    PWMMessage.msg
    SdoWrite.msg
    SetOffset.msg
    Statusword.msg
    Condition.msg
    SppInfo.msg
    ClosedLoopRec.msg
    ControllerDebug.msg
    SyncProfilePositions.msg
    VelocityControl.msg
    VelocityControlFeedback.msg
)

add_service_files(
  FILES
   SdoUpload.srv
   ResetDevice.srv
)

add_action_files(
  FILES
   DiffDriveProfilePosition.action
   Homing.action
   IOLinkDeviceGetOpMode.action
   IOLinkDeviceSetOpMode.action
   IOLinkDeviceGetCfgWord.action
   IOLinkDeviceSetCfgWord.action
   ProfilePosition.action
   ProfileTorque.action
   SyncProfilePosition.action
)

generate_dynamic_reconfigure_options(
  cfg/pid.cfg
)

generate_messages(
  DEPENDENCIES
   actionlib_msgs
   sensor_msgs
   std_msgs)

catkin_package(
  CATKIN_DEPENDS
   actionlib_msgs
   message_runtime
   geometry_msgs
   sensor_msgs
   dynamic_reconfigure
   std_msgs)

install(DIRECTORY launch
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        FILES_MATCHING PATTERN "*.launch")

#install(DIRECTORY params
#        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

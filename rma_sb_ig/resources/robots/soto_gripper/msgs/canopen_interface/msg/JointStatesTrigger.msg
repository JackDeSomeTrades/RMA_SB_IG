# Items to set up trigger
int32 trigger_id              # trigger id to allow multiple trigger setup
sensor_msgs/JointState[] lower_limits  # consists of lower position limit to trigger on
sensor_msgs/JointState[] upper_limits  # consists of upper position limit to trigger on

# Actions to perform after trigger was activated
ProfilePosition[] positions           # Profile Position to start after trigger
SyncProfilePositions[] sync_positions # Syncronized Profile Position to start after trigger
ProfileTorque[] torques               # Profile Torque command to start after trigger
IOMessage[] outputs                   # Outputs to set after trigger
OperationMode[] operation_modes       # operation modes to set after trigger

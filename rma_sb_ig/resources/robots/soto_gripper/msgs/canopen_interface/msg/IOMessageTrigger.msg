# Items to set up trigger
int32     trigger_id                  # id of the trigger, needs to be set for clear, too
IOMessage input_bitmask               # mask bits of input device as trigger
IOMessage input_value                 # value of the masked bits to activate the trigger

# Actions to perform after trigger was activated
ProfilePosition[] positions           # Profile Position to start after trigger
SyncProfilePositions[] sync_positions  # Syncronized Profile Position to start after trigger
ProfileTorque[] torques               # Profile Torque command to start after trigger
IOMessage[] outputs                   # Outputs to set after trigger
OperationMode[] operation_modes       # operation modes to set after trigger

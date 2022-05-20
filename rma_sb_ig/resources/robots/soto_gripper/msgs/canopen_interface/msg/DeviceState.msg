uint8 UNKNOWN = 0       # Unknown state, during the initialization of the node
uint8 UNINITIALIZED = 1 # Device is not initialized yet.
                        # It is equivalent to False in is_initialized
uint8 INITIALIZING = 2  # The device is during the initialization process.
                        # In is_initialized, this time also shows the device
                        # as False
uint8 INITIALIZED = 3   # The device is initialized and functional.
                        # Equivalent to is_initialized as True

time stamp
string[] device_name
string[] base_type
bool[] is_initialized
uint8[] initialization_state

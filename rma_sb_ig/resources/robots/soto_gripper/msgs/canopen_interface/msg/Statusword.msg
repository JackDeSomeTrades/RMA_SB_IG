time stamp
string motor_name
uint16 status
int8 operation_mode
string motor_state

# split of status-word into more readable variables
bool target_reached
bool in_quick_stop
bool fault
bool warning
bool following_error
bool setpoint_ack
bool is_homing
bool homing_finished
bool was_homed
bool is_initialized

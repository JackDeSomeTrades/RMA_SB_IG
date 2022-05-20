#  Constants specifying the condition type
uint8 LESS=0          # is position < value?
uint8 GREATER=1       # is position > value?
uint8 APPROXIMATE=2   # is position > value - epsilon and < value + epsilon?
uint8 AT_TARGET=3      # is position > last_target - epsilon and < last_target + epsilon?

string joint_name
float64 value
uint8 type
float64 epsilon


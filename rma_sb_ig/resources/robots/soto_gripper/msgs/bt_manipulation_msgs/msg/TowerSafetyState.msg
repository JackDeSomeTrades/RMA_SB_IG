# Safety filter is in localization_confirmed state
# when localization of the robot has been cross checked
# (e.g. with shelf markers or charging stations)
bool localization_confirmed

# Remaining `safe` distance the robot is allowed to drive until localization
# is unconfirmed again and current_allowed_height is reset to minimum.
# Strictly positive if localization_confirmed is true, zero if it is false
float32 remaining_distance

# Current maximum allowed joint value of tower_z allowed by the safety filter
# Without localization confirmation, it is either the height of the global
# lowest ceiling obstacle or max robot height allowed by the URDF whichever is
# lower
# With localization confirmation, it is the current_reachable_heigth
float32 current_allowed_height

# The maximum height the robot can reach safely based on where its location is
# thought to be on a map
# It can either be the height of the ceiling obstacle at the current location
# or max robot height allowed by the URDF whichever is lower
float32 current_reachable_height

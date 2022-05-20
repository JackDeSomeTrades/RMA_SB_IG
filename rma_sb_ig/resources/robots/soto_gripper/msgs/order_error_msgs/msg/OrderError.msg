uint8 error          # Order error id, one of the constants
string description   # ID of the blocking object to be repositioned

###############################################################################
#  Availible Error Codes for a full documentation:                            #
#  https://magazino.atlassian.net/wiki/spaces/SD/pages/493617194/Job+problems #
###############################################################################

# Constants defining the error that could occur
uint8 NO_ERROR = 0                              # Everything is fine
uint8 ERROR_COMPARTMENT_EMPTY = 1               # Compartment is empty, nothing to pick
uint8 ERROR_COMPARTMENT_FULL = 2                # Compartment is full, unable to putdown?
uint8 ERROR_OBJECT_SIZE_WRONG = 3               # Object size is unexpected
uint8 ERROR_OBJECT_NOT_FOUND = 4                # Unable to locate the object, ie: Scene too complex
uint8 ERROR_OBJECT_UNREACHABLE = 5              # Unable to grasp an object given its position (stack too high, too far away, behind pillar)
uint8 ERROR_OBJECT_IN_EMPTY_COMPARTMENT = 6     # Object found in an compartment that was supposed to be empty
uint8 UNKNOWN_ERROR = 7                         # Unknown
uint8 ERROR_OBJECT_CANNOT_BE_MANIPULATED = 8    # Unable to manipulate the object given its dimensions
uint8 ERROR_OBJECT_CANNOT_BE_TRANSPORTED = 9    # Unable to transport the object given its dimensions
uint8 ERROR_STACK_CANNOT_BE_MANIPULATED = 10    # There is a stack on top of the object which we cannot handle
uint8 ERROR_WRONG_ITEM_IN_BACKPACK = 11         # The wrong item has been picked by the robot and put in the backpack

# Constants defining the job problem that should be raised
uint8 PROBLEM_ITEM_DOES_NOT_FIT_IN_TARGET = 2
uint8 PROBLEM_ITEM_SIZE_WRONG = 3
uint8 PROBLEM_ITEM_NOT_FOUND = 4
uint8 PROBLEM_ITEM_UNREACHABLE = 5
uint8 PROBLEM_UNKNOWN = 7
uint8 PROBLEM_ITEM_CANNOT_BE_MANIPULATED = 8
uint8 PROBLEM_ITEM_CANNOT_BE_TRANSPORTED = 9
uint8 PROBLEM_ITEM_STACK_CANNOT_BE_MANIPULATED = 10
uint8 PROBLEM_WRONG_ITEM_IN_BACKPACK = 11
uint8 PROBLEM_MALFORMED_JOB = 12
uint8 PROBLEM_PATH_TO_TARGET_BLOCKED = 13
uint8 PROBLEM_CANNOT_CHARGE = 14
uint8 PROBLEM_PUTDOWN_POSITION_UNREACHABLE = 15
uint8 PROBLEM_ITEM_SIZE_OUTSIDE_CONSTRAINTS = 16
uint8 PROBLEM_NO_FREE_TARGET = 17                    # Used by the batcher only
uint8 PROBLEM_MANUAL_EXECUTION_REQUESTED = 18        # Used by the batcher only
uint8 PROBLEM_CONTAINER_UNREACHABLE = 19
uint8 PROBLEM_BATTERY_TOO_LOW = 20
uint8 PROBLEM_CANNOT_DOCK = 21
uint8 PROBLEM_DOCK_SHAPE_NOT_FOUND = 22
uint8 PROBLEM_CANNOT_UNLOCK = 23


# General errors
uint8 NO_ERROR = 0
uint8 UNKOWN = 1

# Execution errors
uint8 ERROR_ITEM_CANNOT_BE_MANIPULATED = 2		# cannot suck object
uint8 ERROR_ITEM_LOST_DURING_MANIPULATION = 3	# sucked, but not lost during pulling in
uint8 ERROR_MOVEMENT_UNSAFE = 4					# e.g. because of safety filter
uint8 ERROR_OVERCURRENT = 5						# collision happened
uint8 ERROR_UNEXPECTED_MANIPULATOR_STATE = 6	# e.g. object in gripper before pickup
uint8 ERROR_UNRESPONSIVE_MOTORS = 7				# could not move the motors
uint8 ERROR_UNRESPONSIVE_SENSORS = 8			# no current sensor information available
uint8 ERROR_WRONG_ITEM_IN_MANIPULATOR = 9       # Gripper blocked by item which is most likely not the target
uint8 ERROR_MEASUREMENT_FAILED = 10             # measuring an item dimensions failed
uint8 ERROR_CONFLICTING_MEASUREMENT = 11        # segmentation & gripper sensors contradict and execution cannot continue as planned

# Planning errors
uint8 ERROR_CANNOT_UNSTACK = 20					# cannot execute the pushback
uint8 ERROR_ITEM_OUTSIDE_SPECS = 21				# item outside the item constraints
uint8 ERROR_REPOSITION_FAILED_TOO_OFTEN = 22	# e.g. end of shelf loop
uint8 ERROR_TARGET_UNREACHABLE = 23				# e.g. object behind pillar, target too far away
uint8 ERROR_WRONG_INPUT = 24					# e.g. rotated object, object not in shelf

uint8 code
string description

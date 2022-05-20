cmake_minimum_required(VERSION 2.8.3)
project(order_error_msgs)

set(MSG_DEPS
  actionlib_msgs
  geometry_msgs
  moveit_msgs
  sensor_msgs
  std_msgs
  visualization_msgs
)

find_package(catkin REQUIRED COMPONENTS ${MSG_DEPS} message_generation)

add_message_files(DIRECTORY msg)
#add_action_files(DIRECTORY action)
#add_service_files(DIRECTORY services)

generate_messages(DEPENDENCIES ${MSG_DEPS})
catkin_package(CATKIN_DEPENDS ${MSG_DEPS} message_runtime)

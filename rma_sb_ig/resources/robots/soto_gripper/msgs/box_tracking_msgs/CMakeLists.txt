cmake_minimum_required(VERSION 2.8.3)
project(box_tracking_msgs)

set(MSG_DEPS
  geometry_msgs
  moveit_msgs
  std_msgs
)

find_package(catkin REQUIRED COMPONENTS ${MSG_DEPS} message_generation)

add_action_files(DIRECTORY action)

generate_messages(DEPENDENCIES ${MSG_DEPS})
catkin_package(CATKIN_DEPENDS ${MSG_DEPS} message_runtime)

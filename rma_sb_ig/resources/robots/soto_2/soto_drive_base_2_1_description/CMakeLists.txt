cmake_minimum_required(VERSION 3.0.2)
project(soto_drive_base_2_1_description)

find_package(catkin REQUIRED
  COMPONENTS
    hlls_build_utilities
)

catkin_package()
save_package_version_yaml()

install(
  DIRECTORY config meshes urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

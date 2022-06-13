cmake_minimum_required(VERSION 3.0.2)
project(soto_backpack_5_1_description)

find_package(catkin REQUIRED
  COMPONENTS
    hardware_configuration_generator
    hlls_build_utilities
)

catkin_package()
generate_urdf_yaml()
save_package_version_yaml()

install(
  DIRECTORY config meshes meta_config urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

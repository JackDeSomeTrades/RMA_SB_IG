cmake_minimum_required(VERSION 2.8.3)

project(neobotix_pcw_description)

find_package(catkin REQUIRED)

catkin_package()

foreach(dir config meshes urdf)
	install(DIRECTORY ${dir}/
		DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)

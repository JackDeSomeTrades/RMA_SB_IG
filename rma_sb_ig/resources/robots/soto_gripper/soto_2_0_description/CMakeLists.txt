cmake_minimum_required(VERSION 2.8.3)
project(soto_2_0_description)

find_package(catkin REQUIRED COMPONENTS roslaunch)

catkin_package()

foreach(dir config meshes urdf)
	install(DIRECTORY ${dir}/
		DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)
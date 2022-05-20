cmake_minimum_required(VERSION 2.8.3)

project(klt_press_1_0_description)

find_package(catkin REQUIRED)

catkin_package()

catkin_package()


foreach(dir config meshes meta_config urdf)
	install(DIRECTORY ${dir}/
		DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)

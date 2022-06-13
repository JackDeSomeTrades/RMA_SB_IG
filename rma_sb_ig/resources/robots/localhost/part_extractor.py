import xml.etree.ElementTree as ET
tree = ET.parse('soto2.urdf')
root = tree.getroot()

def remove_joint_or_link(tree,name,type = "link") :

    if type == "joint" :
        l_to_rm = []
        for child1 in tree :
            if child1.tag == "joint" :
                for prop in child1 : 
                    if prop.tag == "parent" and prop.attrib['link'] == name :
                        for prop2 in child1 :
                            if prop2.tag == "child" :
                                l_to_rm.append(prop2.attrib['link'])
                        print("removing : " + str(child1.attrib["name"]) + " joint " )
                        tree.remove(child1)
        for l in l_to_rm :
            remove_joint_or_link(tree,l,"link")
    elif type == "firstlink" :
        for child3 in tree :
            if child3.tag == "joint" :
                for prop in child3 : 
                    if prop.tag == "child" and prop.attrib['link'] == name :
                        print("removing : " + str(child3.attrib["name"]) + " joint ")
                        tree.remove(child3)
                        break
        remove_joint_or_link(tree,name)

    elif type == "link" : 
        for child2 in tree :
            if child2.tag == "link" and child2.attrib["name"] == name :
                remove_joint_or_link(tree,name,"joint")
                print("removing : " + str(child2.attrib["name"]) )
                tree.remove(child2)


    return tree
        

l_to_remove = ["charger_connector","backpack_light_curtain_emitter_link","backpack_light_curtain_receiver_link","backpack_tilt_link","nav_3d_cam_rear_link", "design_elements_link", "base_front_laser_link", "laser_field_link","base_rear_laser_left_link","base_rear_laser_right_link"]
l_to_remove += ["backpack_layer_0_link","backpack_layer_1_link","backpack_layer_2_link", "backpack_layer_3_link"]
l_to_remove += ["press_base_link","backpack_base_link"]
l_to_remove +=["pcw_bl","pcw_br","pcw_fl","pcw_fr"]
l_to_remove += ["gripper_casing_link","gripper_depth_camera_link"]
l_to_remove += ["light_curtain_emitter_front_link","press_depth_camera_link","light_curtain_emitter_left_link","light_curtain_emitter_right_link","light_curtain_receiver_front_link", "light_curtain_receiver_left_link","light_curtain_receiver_right_link","nav_3d_cam_front_link"]
l_to_remove += ["caster_front_left_base_link", "caster_rear_right_base_link","imu_link","laser_scanner_base_front_link","laser_scanner_base_rear_left_link","laser_scanner_base_rear_right_link","pcw_front_right_base_link","pcw_rear_left_base_link"]
for part_to_rm in l_to_remove :
    remove_joint_or_link(root,part_to_rm,"firstlink")
tree.write("soto_gripper.urdf")

import os
import numpy as np
import xml.etree.ElementTree as ET



NUMBER_OF_LEGS = [3, 4, 5, 6, 7]


class LegXacro:
    leg_tag = "xacro:leg"
    leg_attribute = {name: 'l1', reflect: '1'}
    child_tag = 'origin'
    child_attribute = {xyz: "0.5 -0.1 .75", rpy: "0 0 0"}


def modify_xacro(root):
    child = ET.Element("xacro:leg", )
    # for child in root:
    #     tag = child.tag
    #     attrib = child.attrib
    #     print("tag:", tag, "attrib:", attrib)
    
    

def gen_xacros(num_legs, input_template):
    f = open(input_template, 'a')


if __name__ == '__main__':
    xacro_template = '/home/pavan/Workspace/RMA_SB_IG/rma_sb_ig/resources/robots/v0variants/urdf/v0_template.urdf.xacro'
    tree = ET.parse(xacro_template)
    root = tree.getroot()
    modify_xacro(root)
    for num_legs in NUMBER_OF_LEGS:
        gen_xacros(num_legs)
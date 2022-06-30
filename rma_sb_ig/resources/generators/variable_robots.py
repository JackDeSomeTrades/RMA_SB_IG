import os
import numpy as np
# import xml.etree.ElementTree as ET
import lxml.etree as ET
from pathlib import Path
import subprocess
import random
from pathlib import Path


NUMBER_OF_LEGS = [3, 4, 5, 6, 7]


class VariableRobotDescriptor:
    def __init__(self, xacro_template_fpath:str, xacro_gen_folderpath:str, gen_urdf_folderpath:str, config_dict:dict):
        """
        :param config_dict: dict
        Contains the following attributes:
        "robotDef": dict { num_legs: int}
        "modifyLegs": boolean
        "modifyNbLegs": boolean


        :param xacro_template_fpath: str
        Contains the file path for the xacro template to be modified.
        """
        self.conf_dict = config_dict
        self.xacrons = "http://www.ros.org/wiki/xacro"

        self.fpath_xacro_template = xacro_template_fpath
        self.fpath_xacro_gen = Path(xacro_gen_folderpath)
        self.fpath_urdf_gen = Path(gen_urdf_folderpath)

        parser = ET.XMLParser(remove_blank_text=True)
        self.tree = ET.parse(self.fpath_xacro_template, parser=parser)
        self.root = self.tree.getroot()
        self.ns = {'xacro': self.xacrons}
        self._get_body_size()

    def _get_body_size(self):
        for link in self.root.findall('link'):
            for child in link.iter():
                if 'box' in child.tag:
                    self.sizex, self.sizey = float(child.attrib['sizeX']), float(child.attrib['sizeY'])
    
    def _stringify_value(self, tuple):
        # takes tuple of (x: float, y: float, z: float) as input and converts it into a string 'x y z'
        return f'{tuple[0]} {tuple[1]} {tuple[2]}'

    def _fix_leg_positions(self, x_pos_list):
        x_pos_list = sorted(x_pos_list)
        if len(x_pos_list) > 1:
            for i, pos in enumerate(x_pos_list):
                try:
                    next_pos = x_pos_list[i+1]
                except IndexError:
                    next_pos = x_pos_list[0]

                abs_dist = abs(pos - next_pos)

                if abs_dist < 0.2:
                    if (pos - 0.1) > self.sizex/2:
                        x_pos_list[i] = pos
                        x_pos_list[i+1] = next_pos + 0.15
                    else:
                        x_pos_list[i] = pos - 0.1
                        x_pos_list[i + 1] = next_pos + 0.1

        else:
            pass

        return x_pos_list

    def generate_legs(self):
        # randomly select x position to change for the legs to be modified on the longer side of the body. Random
        # generate positions on y if legs need to be on the shorter side. Generate xacro with reflect -1 to have legs
        # reflect on each side of the axis, for symmetry.
        # origin 0 is at the center of the robot body, size covers the span [-size_ /2., +size_ /2.]
        nblegs_robot = self.conf_dict['robotDef']['num_legs']
        legspos_original = []
        self.legspos = [[[], [], []] for i in range(nblegs_robot)]
        for i in range(nblegs_robot):
            x_pos = np.random.uniform(low=(-self.sizex/2.) + 0.05, high= (self.sizex/2.)-0.05)
            y_pos = np.random.choice([-0.1, 0.1])
            z_pos = 0.75

            legspos_original.append([x_pos, y_pos, z_pos])

        # check if generated leg positions on x are all atleast 0.2 away from each other.
        xpos_on_y_left = [x_pos for x_pos, y_pos, _ in legspos_original if y_pos == 0.1]
        xpos_on_y_right = [x_pos for x_pos, y_pos, _ in legspos_original if y_pos == -0.1]

        xpos_on_y_left = self._fix_leg_positions(xpos_on_y_left)
        xpos_on_y_right = self._fix_leg_positions(xpos_on_y_right)

        left_leg_count = len(xpos_on_y_left)
        right_leg_count = len(xpos_on_y_right)
        llc = 0
        rlc = 0

        for i, legs in enumerate(legspos_original):
            if legs[1] == 0.1 and left_leg_count > 0:
                self.legspos[i][0] = xpos_on_y_left[llc]
                llc += 1
            if legs[1] == -0.1 and right_leg_count > 0:
                self.legspos[i][0] = xpos_on_y_right[rlc]
                rlc += 1
            self.legspos[i][1] = legs[1]
            self.legspos[i][2] = legs[2]

        print("done")

    def write_xacro(self, variant_name):
        # make xml elements for xacro:leg
        for val, legpos in enumerate(self.legspos):
            if val % 2 == 0:
                reflect_val = 1
                leg_prefix = 'l'
            else:
                reflect_val = -1
                leg_prefix = 'r'

            leg_template = ET.Element('{'+f'{self.xacrons}'+'}'+'leg', attrib={'name': f'{leg_prefix}{val+1}', 'reflect': f'{reflect_val}'}, nsmap=self.ns)
            leg_desc = ET.SubElement(leg_template, 'origin', attrib={'xyz': self._stringify_value(legpos), 'rpy': '0 0 0'})
            self.root.append(leg_template)

        if self.fpath_xacro_gen is not None:
            self.fpath_xacro_gen = self.fpath_xacro_gen.joinpath(variant_name).with_suffix('.xacro')
            fl = open(self.fpath_xacro_gen, 'wb')
            # self.tree.write(fl, xml_declaration=True, pretty_print=True)
            fl.write(ET.tostring(self.root, xml_declaration=True, pretty_print=True, with_tail=False))
            fl.close()
            print("Wrote generated Xacro")
    
    def make_urdf(self, variant_name):
        self.fpath_urdf_gen = self.fpath_urdf_gen.joinpath(variant_name).with_suffix('.urdf')
        # env = os.environ
        # env["PATH"] += os.pathsep + '/opt/ros/melodic/bin'     # for the various package not found errors due to path issues
        # env["LD_LIBRARY_PATH"] = '/opt/ros/melodic/lib'
        # env["CMAKE_PREFIX_PATH"] = '/opt/ros/melodic'
        subprocess.run(f'rosrun xacro xacro {self.fpath_xacro_gen}>{self.fpath_urdf_gen}', shell=True)

        print("Finished generating urdf")

    def create(self, variant_name):
        self.generate_legs()
        self.write_xacro(variant_name)
        self.make_urdf(variant_name)


if __name__ == '__main__':
    xacro_template = '/home/pavan/Workspace/RMA_SB_IG/rma_sb_ig/resources/robots/v0variants/v0_template.urdf.xacro'
    xacrogen_folder = '../robots/v0variants/xacros/'
    urdfgen_folder = '../robots/v0variants/urdf/'

    configDict = {
        'robotDef': {'num_legs': 4},
        'modifyLegs': True,
        'modifyNbLegs': False
    }

    NB_GENERATED = 32

    for i in range(NB_GENERATED):
        gen_flname = f'v0_{configDict["robotDef"]["num_legs"]}l_{i}'
        print(f"Generating variant:{i}")
        variance = VariableRobotDescriptor(xacro_template, xacrogen_folder, urdfgen_folder, config_dict=configDict)
        variance.create(gen_flname)


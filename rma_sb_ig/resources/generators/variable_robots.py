import os
import numpy as np
# import xml.etree.ElementTree as ET
import lxml.etree as ET
from pathlib import Path
import subprocess
import random



NUMBER_OF_LEGS = [3, 4, 5, 6, 7]


class VariableRobotDescriptor:
    def __init__(self, fpath: str, config_dict: dict):
        """
        :param config_dict: dict
        Contains the following attributes:
        "robotDef": dict { num_legs: int}
        "modifyLegs": boolean
        "modifyNbLegs": boolean


        :param fpath: str
        Contains the file path for the xacro template to be modified.
        """
        self.conf_dict = config_dict
        self.xacrons = "http://www.ros.org/wiki/xacro"

        self.fpath_xacro = fpath
        parser = ET.XMLParser(remove_blank_text=True)
        self.tree = ET.parse(self.fpath_xacro, parser=parser)
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

    def generate_legs(self):
        # randomly select x position to change for the legs to be modified on the longer side of the body. Random
        # generate positions on y if legs need to be on the shorter side. Generate xacro with reflect -1 to have legs
        # reflect on each side of the axis, for symmetry.
        # origin 0 is at the center of the robot body, size covers the span [-size_ /2., +size_ /2.]
        nblegs_robot = self.conf_dict['robotDef']['num_legs']
        self.legspos = []
        for i in range(nblegs_robot):
            x_pos = np.random.uniform(low=-self.sizex/2., high= self.sizex/2.)
            y_pos = np.random.choice([-0.1, 0.1])
            z_pos = 0.75

            self.legspos.append((x_pos, y_pos, z_pos))
        print("doen")

    def write_xacro(self, fpath:str):
        # make xml elements for xacro:leg
        for val, legpos in enumerate(self.legspos):
            reflect_val = random.choice([-1, 1])
            leg_template = ET.Element('{'+f'{self.xacrons}'+'}'+'leg', attrib={'name': f'l{val+1}', 'reflect': f'{reflect_val}'}, nsmap=self.ns)
            leg_desc = ET.SubElement(leg_template, 'origin', attrib={'xyz': self._stringify_value(legpos), 'rpy': '0 0 0'})
            # ET.dump(leg_template)

            self.root.append(leg_template)

        if fpath is not None:
            fl = open(fpath, 'wb')
            # self.tree.write(fl, xml_declaration=True, pretty_print=True)
            fl.write(ET.tostring(self.root, xml_declaration=True, pretty_print=True, with_tail=False))
            fl.close()
    
    def make_urdf(self, outpath):
        cmd = ["rosrun", "xacro", "xacro", f"{self.fpath_xacro} > {outpath}"]
        # subprocess.run(cmd, env=os.environ, stdout=subprocess.PIPE, check=True)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=os.environ, preexec_fn=os.setsid())
        print("Finished generating urdf")


if __name__ == '__main__':
    xacro_template = '/home/pavan/Workspace/RMA_SB_IG/rma_sb_ig/resources/robots/v0variants/urdf/v0_template.urdf.xacro'

    configDict = {
        'robotDef': {'num_legs': 4},
        'modifyLegs': True,
        'modifyNbLegs': False
    }

    variance = VariableRobotDescriptor(xacro_template, config_dict=configDict)
    variance.generate_legs()
    variance.write_xacro('/home/pavan/Workspace/RMA_SB_IG/rma_sb_ig/resources/robots/v0variants/urdf/v0_4_legs_generated.urdf.xacro')
    variance.make_urdf('/home/pavan/Workspace/RMA_SB_IG/rma_sb_ig/resources/robots/v0variants/urdf/v0_4_legs_generated.urdf')

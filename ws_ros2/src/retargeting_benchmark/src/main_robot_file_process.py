import os
import xml.etree.ElementTree as ET
from math import sqrt

import numpy as np
from utils.utils_mjcf import (
    array_to_string,
    find_elements,
    find_parent,
    indent,
    nested_mjcf_to_single_mjcf,
    new_element,
    string_to_array,
)
from utils.utils_urdf import urdf_to_xml

"""
1. Convert the urdf file to xml file.
2. Add taxel (site array) to the fingertips.
3. Save the new xml file.
"""

if __name__ == "__main__":
    # hyper-parameters for Tac3D sensor
    gravcomp: bool = True
    ignore_collision_of_fingertip: bool = False
    fingertip_names = [
        "thumb_fingertip_new",
        "fingertip_new",
        "fingertip_2_new",
        "fingertip_3_new",
    ]
    fingertip_center_names = [
        "thumb_tip_center",
        "finger1_tip_center",
        "finger2_tip_center",
        "finger3_tip_center",
    ]

    # ------------------------------------------------------------
    # first, convert the urdf to xml
    urdf_path = os.readlink("assets/panda_leap_tac3d.urdf")  # using soft link
    xml_path = "assets/panda_leap_tac3d.xml"
    urdf_to_xml(urdf_path, xml_path)
    print("Convert the urdf file to xml file.")

    # ------------------------------------------------------------
    # robot body xml file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # add name to the fingertip geom
    # add 'class' attribute to the panda
    geoms = find_elements(root, tags="geom", return_first=False)
    for geom in geoms:
        body_name = find_parent(root, geom).get("name")
        if (body_name in fingertip_names) and ("contype" not in geom.attrib):
            geom.set("name", f"{body_name}_g")
            if ignore_collision_of_fingertip:
                geom.set("contype", "0")
                geom.set("conaffinity", "0")

    print("Add 'name' attribute to the fingertip geoms.")

    # add 'class' attribute to the panda
    joints = find_elements(root, tags="joint", return_first=False)
    for joint in joints:
        if joint.get("name").startswith("panda_"):
            joint.set("class", "panda")
    print("Add 'class' attribute to the Panda.")

    # add 'gravcomp' to the robot bodies
    if gravcomp:
        bodies = find_elements(root, tags="body", return_first=False)
        for body in bodies:
            body.set("gravcomp", "1")
        print("Add 'gravcomp = 1' attribute to the robot.")

    # save the xml file
    indent(root)  # format the xml
    save_path = "assets/panda_leap_tac3d.xml"
    tree.write(save_path, encoding="utf-8")
    print(f"Save {save_path}.")

    print("------ Finished all ------")

import numpy as np
import mujoco_py
from mujoco_py import MjSim, MjViewer

# Path to the XML file containing the UR5e robot arm model
robot_path = "/home/branko/Desktop/medical robotics/ur5e.xml"

# Read the UR5e robot XML content
with open(robot_path, 'r') as f:
    robot_xml = f.read()

# Define the XML model including the robot arm
xml_model = f"""
<mujoco>
    <compiler angle="radian" meshdir="assets" autolimits="true"/>
    <option integrator="implicitfast"/>

    <asset>
        <material name="floor_material" rgba="0.5 0.7 1.0 1"/> <!-- Light blue -->
        <material name="sky_material" rgba="0.1 0.5 0.9 1"/>
    </asset>

    <worldbody>
        <!-- Insert the UR5e robot arm here -->
        {robot_xml}

        <!-- Define the ellipsoid body -->
        <body name="ellipsoid_body" pos="0.8 0.2 0.11">
            <joint name="ellipsoid_joint" type="hinge" axis="1 0 0"/>
            <geom type="ellipsoid" size="0.15 0.1 0.05" rgba="1 0 0 1"/>
        </body>

        <!-- Define the floor -->
        <geom name="floor" type="plane" size="20 20 0.1" material="floor_material" pos="0 0 0"/>

        <!-- Define the sky -->
        <geom name="sky" type="plane" size="20 20 0.1" material="sky_material" pos="0 0 5"/>

        <!-- Define lights -->
        <light name="light1" pos="0 0 5" dir="0 0 -1" diffuse="1 1 1" specular="1 1 1"/>
    </worldbody>

    <option gravity="0 0 0"/> <!-- Disable gravity -->
</mujoco>
"""

# Load the combined model
combined_model = mujoco_py.load_model_from_xml(xml_model)
sim = MjSim(combined_model)
viewer = MjViewer(sim)

joint_id = sim.model.joint_name2id('ellipsoid_joint')

while True:
    t = sim.data.time  
    sway_amplitude = 0.05  
    sway_frequency = 0.5  
    sway_offset = 0.1  
    sway = sway_amplitude * np.sin(2 * np.pi * sway_frequency * t) + sway_offset
    sim.data.qpos[joint_id] = sway
    
    sim.step()
    viewer.render()

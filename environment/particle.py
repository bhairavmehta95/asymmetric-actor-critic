#!/usr/bin/env python3

from mujoco_py import load_model_from_xml, MjSim, MjViewer
import math
import os


MODEL_XML = """
<?xml version="1.0" ?>
<mujoco>
    <option timestep="0.005" />
    <worldbody>
        <geom name="floor" type="plane" size="2 2 2" pos="0 0 0"/>
        <geom type="plane" size=".1905 .2667 .005"/>
        <body name="agent" pos="0 0 0.055">
            <joint axis="1 0 0" damping="0.1" name="slide0" pos="0 0 0" type="slide"/>
            <joint axis="0 1 0" damping="0.1" name="slide1" pos="0 0 0" type="slide"/>
            <geom mass="1.0" pos="0 0 0" rgba="1 0 0 1" size="0.1" type="sphere"/>
        </body>
    </worldbody>
    <actuator>
        <motor gear="2000.0" joint="slide0"/>
        <motor gear="2000.0" joint="slide1"/>
    </actuator>
</mujoco>
"""

model = load_model_from_xml(MODEL_XML)
sim = MjSim(model)
viewer = MjViewer(sim)
viewer.cam.trackbodyid = 1
viewer.cam.elevation = -89
viewer.cam.distance = 7.05
viewer.cam.lookat[1] += 1
t = 0
while True:
    sim.data.ctrl[0] = math.cos(t / 10.) * 0.1
    sim.data.ctrl[1] = math.sin(t / 10.) * 0.1
    t += 1
    sim.step()
    viewer.render()
    print(viewer.cam.distance)
    if t > 100 and os.getenv('TESTING') is not None:
        break
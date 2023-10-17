import os
import numpy as np
import pybullet as p
import pybullet_data
import time


def pybullet_gui_test():
    p.connect(p.GUI)

    urdfRootPath=pybullet_data.getDataPath()

    p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"),useFixedBase=True)
    p.loadURDF(os.path.join("./urdf/bed/bed.urdf"),basePosition=[0.6,-0.2,-0.65])
    p.loadURDF(os.path.join("./urdf/human/human_2.urdf"),basePosition=[0.6,0,0.2], baseOrientation=[-0.5,-0.5,-0.5,0.5])
    p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),basePosition=[-0.65, 0, -0.5])
    p.loadURDF(os.path.join(urdfRootPath, "objects/mug.urdf"),useFixedBase=False, basePosition=[-0.55, 0, 0.13])
    p.loadURDF(os.path.join("./urdf/roller/roller.urdf"),useFixedBase=False, basePosition=[-0.5, 0.2, 0.15], baseOrientation=[1,0,1,0])
    p.loadURDF(os.path.join("./urdf/fork/fork.urdf"),useFixedBase=False, basePosition=[-0.4, -0.4, 0.15], baseOrientation=[1,0,1,0])
    p.loadURDF(os.path.join("./urdf/bottle/bottle.urdf"),useFixedBase=False, basePosition=[-0.4, -0.2, 0.15], baseOrientation=[1,0,1,0])
    p.loadURDF(os.path.join("./urdf/food/food_item0.urdf"),useFixedBase=False, basePosition=[-0.4, 0.3, 0.15])

    while True:
        p.stepSimulation()
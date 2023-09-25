import os
import numpy as np
import pybullet as p
import pybullet_data
import time


def pybullet_gui_test():
    p.connect(p.GUI)

    urdfRootPath=pybullet_data.getDataPath()

    

    p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"),useFixedBase=True)
    p.loadURDF(os.path.join("./bed/bed.urdf"),basePosition=[0.5,0,-0.65])
    
    while True:
        p.stepSimulation()

    # jointid = 4
    # jlower = p.getJointInfo(pandaUid, jointid)[8]
    # jupper = p.getJointInfo(pandaUid, jointid)[9]

    # # for step in range(300):
    # while True:
    #     joint_two_targ = np.random.uniform(jlower, jupper)
    #     joint_four_targ = np.random.uniform(jlower, jupper)

    #     p.setJointMotorControlArray(pandaUid, [0, 1], p.POSITION_CONTROL, targetPositions = [joint_two_targ, joint_four_targ])
    #     p.stepSimulation()
    #     time.sleep(.01)
import os
import numpy as np
import pybullet as p
import pybullet_data
import time


def pybullet_gui_test():
    p.connect(p.GUI)
    pandaUid = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf"),useFixedBase=True)

    # while True:
    #     p.stepSimulation()

    jointid = 4
    jlower = p.getJointInfo(pandaUid, jointid)[8]
    jupper = p.getJointInfo(pandaUid, jointid)[9]

    for step in range(300):
        joint_two_targ = np.random.uniform(jlower, jupper)
        joint_four_targ = np.random.uniform(jlower, jupper)

        p.setJointMotorControlArray(pandaUid, [2, 4], p.POSITION_CONTROL, targetPositions = [joint_two_targ, joint_four_targ])
        focus_position, _ = p.getBasePositionAndOrientation(pandaUid)
        p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=0, cameraPitch=-40, cameraTargetPosition = focus_position)
        p.stepSimulation()
        time.sleep(.01)
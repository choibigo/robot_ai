import os
import pybullet as p
import pybullet_data
import numpy as np
from vmp import VMP

traj_path = "data/traj_data"

if __name__ == "__main__":
    pandaEndEffectorIndex = 8
    numJoints = 8
    vmp = VMP(3, kernel_num=50, elementary_type='linear', use_out_of_range_kernel=False)
    vmp.load_weights_from_file('circular_fast')
    start = np.array([ -0.5,0.2,0.65])
    goal = np.array([0.6,0,0.9])
    reproduced = vmp.roll(start, goal, 50)
    print(reproduced)

    # region pybullet
    p.connect(p.GUI)
    pandaUid = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf"),useFixedBase=True)
    flag = False
    p.setRealTimeSimulation(1)
    p.setTimeStep(1/120)
    i = 0

    while i<len(reproduced)-1:
        jointPoses = p.calculateInverseKinematics(pandaUid,
                                                 pandaEndEffectorIndex,
                                                 reproduced[i][1:]
                                                 )
        for j in range(numJoints):
            p.setJointMotorControl2(bodyIndex=pandaUid,
                                    jointIndex=j,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[j],
                                    targetVelocity=0,
                                    )
        for k in range(1000):
            p.stepSimulation()
        i += 1
        
    # endregion
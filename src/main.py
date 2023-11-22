import argparse
from simulation.my_env import SimulationEnvironment
from grasping.grasp_detector import GrasppingScenarios
from movement_primitive.vmp import VMP
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--instruction', type=str, default="Give me a meat can", help='Enter your instruction')       
    parser.add_argument('--goal_point', type=str, default="right_arm", help='You have to choose from the candidates')
    parser.add_argument('--style', type=str, default="massage" , help='The name of motion style to train or reproduce')       
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # region Build Enviroment
    env = SimulationEnvironment()
    pybullet_simulation = env.build()
    # endregion

    # region Grasping & Skeleton Detection
    grasp = GrasppingScenarios()   
    goal = grasp.scenario(env, args.instruction, args.goal_point)
    # endregion

    env.move_initial()

    # region Motion Generation
    vmp = VMP()
    vmp.load_weights_from_file(args.style)
    start = np.array([-0.11, 0.4958, 1.0611]) # Check - seonho
    reproduced = vmp.roll(start,goal,50)
    env.mp_control(reproduced)
    # endregion

import argparse
from simulation.my_env import SimulationEnvironment
from grasping.grasp_detector import GrasppingScenarios
from movement_primitive.vmp import VMP
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--instruction', type=str, default="Give me a meat can", help='')       
    parser.add_argument('--goal_point', type=str, default="right_arm", help='')
    parser.add_argument('--style', type=str, default="massage" , help='the name of motion style to train or reproduce')       
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # Build Enviroment
    env = SimulationEnvironment()
    pybullet_simulation = env.build()

    ###      Grasping       ###
    # instruction = "Give me something to cut" 
    # instruction = "I want to eat fruit"   
    # instruction = "Pour the sauce"
    grasp = GrasppingScenarios()   
    goal = grasp.scenario(env, args.instruction, args.goal_point)

    env.move_initial()

    ### Motion Generation  ###
    # viapoint movement primitive
    vmp = VMP(3, kernel_num=50, elementary_type='linear', use_out_of_range_kernel=False)
    vmp.load_weights_from_file(args.style)
    start = np.array([-0.11, 0.4958, 1.0611])
    reproduced = vmp.roll(start,goal,50)
    env.mp_control(reproduced)
    # for _ in range(1000):
    #     env.step_simulation()
    #     end = env.get_eff()
    # print(end)


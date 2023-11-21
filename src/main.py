import argparse
from simulation.my_env import SimulationEnvironment
from grasping.grasp_detector import GrasppingScenarios

def parse_args():
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--instruction', type=str, default="Give me a meat can", help='')       
    parser.add_argument('--goal_point', type=str, default="mouth", help='')
    parser.add_argument('--style', type=str, default="play" , help='the name of motion style to train or reproduce')       
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # Build Enviroment
    env = SimulationEnvironment()
    pybullet_simulation = env.build()

    # while True:
    #     # region Grasping3
    #     pass
    #     # endregion

    #     # region Skeleton Detection
    #     rgb_image, depth_image = env.camera_set(env.camera_1_config)
    #     detection_image, skeleton_info = skeleton_detector.detection(rgb_image)
    #     real_coordinate_from_cramera_image = env.get_point_cloud()
    #     skeleton_detector.show(detection_image)
    #     # endregion

    env.move_initial()

    #     # region Motion Generation
    # # viapoint movement primitive
    # vmp = VMP(3, kernel_num=50, elementary_type='linear', use_out_of_range_kernel=False)
    # vmp.load_weights_from_file(f'{args.style}')    
    # start = np.array([-0.11,0.4958,1.0611])
    # goal = np.array([0.2806,0.0937,1.0095])
    # reproduced = vmp.roll(start,goal,50)
    # env.mp_control(reproduced)
    #     pass
    #     # endregion

    #     pybullet_simulation.stepSimulation()
    
    # Grasping with skeleton detection
    # grasp = GrasppingScenarios()   
    # grasp.scenario(env, args.instruction, args.goal_point)

    # instruction = "Give me something to cut" 
    # instruction = "Feed me something to eat"   
    # instruction = "Pour the sauce"

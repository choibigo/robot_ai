import argparse
from simulation.my_env import SimulationEnvironment
#from grasping.grasp_detector import GrasppingScenarios
from movement_primitive.vmp import VMP
import numpy as np
import csv

# def parse_args():
#     parser = argparse.ArgumentParser(description='Demo')
#     parser.add_argument('--instruction', type=str, default="Give me a meat can", help='')       
#     parser.add_argument('--goal_point', type=str, default="mouth", help='')
#     parser.add_argument('--style', required=True, help='the name of motion style to train or reproduce')       
#     args = parser.parse_args()
#     return args

if __name__ == "__main__":
    #args = parse_args()

    # Build Enviroment
    env = SimulationEnvironment()
    pybullet_simulation = env.build()

    # viapoint movement primitive
    # vmp = VMP(3, kernel_num=50, elementary_type='linear', use_out_of_range_kernel=False)
    # vmp.load_weights_from_file('linear')
    # start = np.array([0.08,0.4,0.7])
    # goal = np.array([0.4,-0.2,0.545])
    # reproduced = vmp.roll(start,goal,50)
    # env.mp_control(reproduced)
    


    for i in range(1000):
        env.get_eff()
        print(env.end)
    traj1_csv_file = 'main_test_massage.csv'

    # Write traj1 data to CSV file
    with open(traj1_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write traj1 data
        writer.writerows(env.end)
    print('success')
    


    
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
        
    #     # region Motion Generation
    #     pass
    #     # endregion

    #     pybullet_simulation.stepSimulation()
    # Grasping with skeleton detection
    #grasp = GrasppingScenarios()   
    #grasp.scenario(env, args.instruction, args.goal_point)

    # instruction = "Give me something to cut" 
    # instruction = "Feed me something to eat"   
    # instruction = "Pour the sauce"
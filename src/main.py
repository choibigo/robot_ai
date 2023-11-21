# First Executable File
from simulation.my_env import SimulationEnvironment
from skeleton.detector import SkeletonDetection
from movement_primitive.vmp import VMP
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--style', required=True,
                        help='the name of motion style to train or reproduce')
    args = parser.parse_args()

    env = SimulationEnvironment()
    skeleton_detector = SkeletonDetection()
    pybullet_simulation = env.build()

    # viapoint movement primitive
    vmp = VMP(3, kernel_num=50, elementary_type='linear', use_out_of_range_kernel=False)
    vmp.load_weights_from_file(f'{args.style}')
    
    while True:
        # region Grasping3
        pass
        # endregion

        # region Skeleton Detection
        rgb_image, depth_image = env.camera_set(env.camera_1_config)
        detection_image, skeleton_info = skeleton_detector.detection(rgb_image)
        real_coordinate_from_cramera_image = env.get_point_cloud()
        skeleton_detector.show(detection_image)
        # endregion
        
        # region Motion Generation
        pass
        # endregion

        pybullet_simulation.stepSimulation()

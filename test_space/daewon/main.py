# First Executable File
from simulation_test.my_env import SimulationEnvironment
from skeleton_test.detector import SkeletonDetection

if __name__ == "__main__":
    env = SimulationEnvironment()
    skeleton_detector = SkeletonDetection()
    pybullet_simulation = env.build()
    
    while True:
        # region Grasping3
        pass
        # endregion

        # region Skeleton Detection
        rgb_image, _ = env.camera_set(env.camera_1_config)
        detection_image, skeleton_info = skeleton_detector.detection(rgb_image)
        real_coordinate_from_cramera_image = env.get_point_cloud()
        skeleton_detector.show(detection_image)
        # endregion
        
        # region Motion Generation
        pass
        # endregion

        pybullet_simulation.stepSimulation()
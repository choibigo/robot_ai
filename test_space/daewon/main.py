# 최초 실행 파일

from simulation_test.my_env import SimulationEnvironment
from skeleton_test.detector import SkeletonDetection

if __name__ == "__main__":
    env = SimulationEnvironment()
    skeleton_detector = SkeletonDetection()
    pybullet_simulation = env.build()

    while True:
        rgb_image, depth_image = env.camera_set(env.camera_1_config)
        detection_image = skeleton_detector.detection(rgb_image)
        skeleton_detector.show(detection_image)
        pybullet_simulation.stepSimulation()
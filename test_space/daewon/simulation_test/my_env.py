import os
import numpy as np
import pybullet as p
import pybullet_data

class SimulationEnvironment:

    def __init__ (self):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.current_path = os.path.dirname(os.path.realpath(__file__))

        # camera 1: Skeleton Detection
        self.camera_1_config = dict()
        self.camera_1_config['width'] = 500
        self.camera_1_config['height'] = 500
        self.camera_1_config['fov'] = 60
        self.camera_1_config['aspect'] = self.camera_1_config['width'] / self.camera_1_config['height'] 
        self.camera_1_config['near'] = 0.02
        self.camera_1_config['far'] = 10
        self.camera_1_config['view_matrix'] = p.computeViewMatrix([0.5, 0, 3.5], [0.5, 0, 0], [1, 0, 0])
        self.camera_1_config['projection_matrix'] = p.computeProjectionMatrixFOV(self.camera_1_config['fov'],
                                                                                 self.camera_1_config['aspect'],
                                                                                 self.camera_1_config['near'],
                                                                                 self.camera_1_config['far'])

        # # camera 2: Grasping (성훈 맞는 위치 찾아야 함)
        # self.camera_2_config = dict()
        # self.camera_2_config['width'] = 500
        # self.camera_2_config['height'] = 500
        # self.camera_2_config['fov'] = 60
        # self.camera_2_config['aspect'] = self.camera_2_config['width'] / self.camera_2_config['height'] 
        # self.camera_2_config['near'] = 0.02
        # self.camera_2_config['far'] = 10
        # self.camera_2_config['view_matrix'] = p.computeViewMatrix([1, 2, 3], [0, 0, 0], [1, 0, 0])
        # self.camera_2_config['projection_matrix'] = p.computeProjectionMatrixFOV(self.camera_2_config['fov'],
        #                                                                          self.camera_2_config['aspect'],
        #                                                                          self.camera_2_config['near'],
        #                                                                          self.camera_2_config['far'])

    def __urdf_build(self):
        p.loadURDF('plane.urdf')
        # p.loadURDF("franka_panda/panda.urdf",useFixedBase=True, basePosition=[0, 0, 0.6])
        p.loadURDF(os.path.join(self.current_path, "urdf/bed/bed.urdf"),basePosition=[0.6,-0.2,0])
        p.loadURDF(os.path.join(self.current_path, "urdf/human/human_2.urdf"),basePosition=[0.6,0,0.9], baseOrientation=[-0.5,-0.5,-0.5,0.5])
        p.loadURDF("table/table.urdf",basePosition=[-0.65, 0, 0])
        p.loadURDF("objects/mug.urdf",useFixedBase=False, basePosition=[-0.55, 0, 0.65])
        p.loadURDF(os.path.join(self.current_path, "urdf/roller/roller.urdf"),useFixedBase=False, basePosition=[-0.5, 0.2, 0.65], baseOrientation=[1,0,1,0])
        p.loadURDF(os.path.join(self.current_path, "urdf/fork/fork.urdf"),useFixedBase=False, basePosition=[-0.4, -0.4, 0.65], baseOrientation=[1,0,1,0])
        p.loadURDF(os.path.join(self.current_path, "urdf/bottle/bottle.urdf"),useFixedBase=False, basePosition=[-0.4, -0.2, 0.65], baseOrientation=[1,0,1,0])
        p.loadURDF(os.path.join(self.current_path, "urdf/food/food_item0.urdf"),useFixedBase=False, basePosition=[-0.4, 0.3, 0.65])

    def camera_set(self, camera_config):
        self.images = p.getCameraImage(camera_config['width'], camera_config['height'], camera_config['view_matrix'], camera_config['projection_matrix'], renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgba = (np.reshape(self.images[2], (camera_config['width'], camera_config['height'], 4)))
        
        depth_image = 'you have to implement depth_image'

        def __rgba2rgb(input_rgba):
            row, col, ch = rgba.shape

            if ch == 3:
                return rgba

            assert ch == 4, 'RGBA image has 4 channels.'

            rgb = np.zeros( (row, col, 3), dtype='float32' )
            r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

            a = np.asarray( a, dtype='float32' ) / 255.0

            R, G, B = (255,255,255)

            rgb[:,:,0] = r * a + (1.0 - a) * R
            rgb[:,:,1] = g * a + (1.0 - a) * G
            rgb[:,:,2] = b * a + (1.0 - a) * B

            return np.asarray( rgb, dtype='uint8' )
        
        return __rgba2rgb(rgba), depth_image

    def get_point_cloud(self):
        depth = self.images[3]
        tran_pix_world = np.linalg.inv(np.matmul(np.asarray(self.camera_1_config['projection_matrix']).reshape([4, 4], order="F"),
                                                 np.asarray(self.camera_1_config['view_matrix']).reshape([4, 4], order="F")))

        # create a grid with pixel coordinates and depth values
        y, x = np.mgrid[-1:1:2 / self.camera_1_config['height'], -1:1:2 / self.camera_1_config['width']]
        y *= -1.
        # x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
        x, y, z = x.reshape(-1), y.reshape(-1), np.asarray(depth)
        h = np.ones_like(z)

        pixels = np.stack([x, y, z, h], axis=1)
        # filter out "infinite" depths
        pixels = pixels[z < 0.99999999999]
        pixels[:, 2] = 2 * pixels[:, 2] - 1

        # turn pixels to world coordinates
        points = np.matmul(tran_pix_world, pixels.T).T
        points /= points[:, 3: 4]
        points = points[:, :3]

        points = points.reshape(self.camera_1_config['width'],
                                self.camera_1_config['height'],
                                3 )

        return points

    def build(self):
        self.__urdf_build()
        return p
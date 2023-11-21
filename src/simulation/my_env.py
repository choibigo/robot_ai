from grasping.environment.utilities import setup_sisbot, Camera
import os
import numpy as np
import pybullet as p
import pybullet_data
import random
import math
import time

class SimulationEnvironment:

    def __init__ (self):
        p.connect(p.GUI)
        # p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.current_path = os.path.dirname(os.path.realpath(__file__))
        p.setTimeStep(1/120)

        cameraDistance = 2.0
        cameraYaw = -50
        cameraPitch = -50
        cameraTargetPosition = [-0.9, -0.7, 1.0]
        p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition)

        self.gripper_open_limit = (0.0, 0.1)
        self.ee_position_limit = ((-0.8, 0.8),
                                (-0.8, 0.8),
                                (0.785, 1.4))
        self.end = np.empty((0,3))

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

        # # camera 2: Grasping
        self.camera_2_config = dict()
        self.camera_2_config['width'] = 300
        self.camera_2_config['height'] = 300
        self.camera_2_config['fov'] = 60
        self.camera_2_config['aspect'] = self.camera_2_config['width'] / self.camera_2_config['height'] 
        self.camera_2_config['near'] = 0.02
        self.camera_2_config['far'] = 10
        self.camera_2_config['view_matrix'] = p.computeViewMatrix([-1, 0, 2.5], [-1, 0, 0], [1, 0, 0])
        self.camera_2_config['projection_matrix'] = p.computeProjectionMatrixFOV(self.camera_2_config['fov'],
                                                                                 self.camera_2_config['aspect'],
                                                                                 self.camera_2_config['near'],
                                                                                 self.camera_2_config['far'])
        


        # Robot Info : eeIndex 확인필요
        self.eeIndex = 7



    def __urdf_build(self):
        p.loadURDF('plane.urdf')
        #p.loadURDF("franka_panda/panda.urdf",useFixedBase=True, basePosition=[0, 0, 0.6])
        self.robot_id = p.loadURDF(os.path.join(self.current_path,"urdf/ur5/ur5_robotiq_140.urdf"),
                                   [0, 0, -0.15],
                                   p.getQuaternionFromEuler([0, 0, math.pi/2]),
                                   useFixedBase=False,
                                   flags=p.URDF_USE_INERTIA_FROM_FILE)
        self.joints, self.controlGripper, self.controlJoints, self.mimicParentName = setup_sisbot(p, self.robot_id, '140')
        self.reset_robot()
        p.loadURDF(os.path.join(self.current_path, "urdf/bed/bed.urdf"),basePosition=[0.6,-0.2,0])
        self.target_id = p.loadURDF(os.path.join(self.current_path, "urdf/human/human_2.urdf"),basePosition=[0.6,0,0.9], baseOrientation=[-0.5,-0.5,-0.5,0.5])
        self.tableID = p.loadURDF(os.path.join(self.current_path,"urdf/table/table.urdf"),
                                  [-0.5, 0, 0.6],
                                  p.getQuaternionFromEuler([0, 0, 0]),
                                  useFixedBase=True)

        self.banana = p.loadURDF(os.path.join(self.current_path, "urdf/YcbBanana/YcbBanana.urdf"),useFixedBase=False, basePosition=[-0.4, 0.18, 0.695], baseOrientation=[0, 0, 0.38268343, 0.92387953])
        self.mustardbottle = p.loadURDF(os.path.join(self.current_path, "urdf/YcbMustardBottle/YcbMustardBottle.urdf"),useFixedBase=False, basePosition=[-0.8, -0.18, 0.71], baseOrientation=[0,1,0,1])
        self.pottedmeatcan = p.loadURDF(os.path.join(self.current_path, "urdf/YcbPottedMeatCan/YcbPottedMeatCan.urdf"),useFixedBase=False, basePosition=[-0.4, -0.18, 0.695], baseOrientation=[0, 0, 0.38268343, 0.92387953])
        self.scissors = p.loadURDF(os.path.join(self.current_path, "urdf/YcbScissors/YcbScissors.urdf"),useFixedBase=False, basePosition=[-0.8, 0.18, 0.53], baseOrientation=[0, 0, 0.38268343, 0.92387953])
        


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
    
    def dummy(self,num):
        for i in range(num):
            self.step_simulation()
    
    def mp_control(self, reproduced):
        for i in range(len(reproduced)-1):
            orn = p.getQuaternionFromEuler([0, np.pi/2, 0.0])
            action = [reproduced[i][1],reproduced[i][2],reproduced[i][3], orn]
            self.move_ee(action)
            for _ in range(40):
                p.stepSimulation()
                
    def get_eff(self):
        time.sleep(0.01)
        p.stepSimulation()
        eef_xyz = p.getLinkState(self.robot_id, 7)[0:1]
        end = np.array([eef_xyz[0]])
        print(end)
        self.end = np.append(self.end,end,axis=0)
        return self.end

    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        p.stepSimulation()

        eef_xyz = p.getLinkState(self.robot_id, 7)[0:1]
        end = np.array(eef_xyz[0])
        end[2] -= 0.5

        time.sleep(0.0001)

    def reset_robot(self):
        user_parameters = (0, -1.5446774605904932, 1.54, -1.54,
                           -1.5707970583733368, 0.0009377758247187636, 0.085)
        for _ in range(180):
            for i, name in enumerate(self.controlJoints):
                
                joint = self.joints[name]
                # control robot joints
                p.setJointMotorControl2(self.robot_id, joint.id, p.POSITION_CONTROL,
                                        targetPosition=user_parameters[i], force=joint.maxForce,
                                        maxVelocity=joint.maxVelocity/10)
                self.step_simulation()
                
            self.controlGripper(controlMode=p.POSITION_CONTROL, targetPosition=0.085)
            self.step_simulation()
    
    def move_gripper(self, gripper_opening_length: float, step: int = 120):
        gripper_opening_length = np.clip( gripper_opening_length, *self.gripper_open_limit)
        gripper_opening_angle = 0.715 - math.asin((gripper_opening_length - 0.010) / 0.1143)  # angle calculation
            
        for _ in range(step):
            self.controlGripper(controlMode=p.POSITION_CONTROL,
                                targetPosition=gripper_opening_angle)

            self.step_simulation()
    
    def check_grasped_id(self):
        left_index = self.joints['left_inner_finger_pad_joint'].id
        right_index = self.joints['right_inner_finger_pad_joint'].id

        contact_left = p.getContactPoints(
            bodyA=self.robot_id, linkIndexA=left_index)
        contact_right = p.getContactPoints(
            bodyA=self.robot_id, linkIndexA=right_index)
        contact_ids = set(item[2] for item in contact_left +
                          contact_right if item[2] in self.obj_ids)
        if len(contact_ids) > 1:
            if self.debug:
                print('Warning: Multiple items in hand!')
        return list(item_id for item_id in contact_ids if item_id in self.obj_ids)

    def gripper_contact(self, bool_operator='and', force=250):
        left_index = self.joints['left_inner_finger_pad_joint'].id
        right_index = self.joints['right_inner_finger_pad_joint'].id

        contact_left = p.getContactPoints( bodyA=self.robot_id, linkIndexA=left_index)
        contact_right = p.getContactPoints( bodyA=self.robot_id, linkIndexA=right_index)

        if bool_operator == 'and' and not (contact_right and contact_left):
            return False

    def move_ee(self, action, max_step=1500, check_collision_config=None, custom_velocity=None,
                try_close_gripper=False, verbose=False):
        x, y, z, orn = action
        x = np.clip(x, *self.ee_position_limit[0])
        y = np.clip(y, *self.ee_position_limit[1])
        z = np.clip(z, *self.ee_position_limit[2])
        # set damping for robot arm and gripper
        jd = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
              0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        jd = jd * 0
        still_open_flag_ = True  # Hot fix

        real_xyz, real_xyzw = p.getLinkState(self.robot_id, 7)[0:2]
        alpha = 0.2 # this parameter can be tuned to make the movement  smoother
        
        for _ in range(max_step):

            # apply IK
            x_tmp = alpha * x + (1-alpha)*real_xyz[0]
            y_tmp = alpha * y + (1-alpha)*real_xyz[1]
            z_tmp = alpha * z + (1-alpha)*real_xyz[2]
            
            joint_poses = p.calculateInverseKinematics(bodyUniqueId=self.robot_id, endEffectorLinkIndex=7, 
                                                       targetPosition=[x_tmp, y_tmp, z_tmp], targetOrientation=orn, 
                                                       maxNumIterations=200)

            # Filter out the gripper
            for i, name in enumerate(self.controlJoints[:-1]):
                joint = self.joints[name]
                pose = joint_poses[i]
                # control robot end-effector
                p.setJointMotorControl2(self.robot_id, joint.id, p.POSITION_CONTROL,
                                        targetPosition=pose, force=joint.maxForce,
                                        maxVelocity=joint.maxVelocity/10)

            self.step_simulation()
            if try_close_gripper and still_open_flag_ and not self.gripper_contact():
                still_open_flag_ = self.close_gripper(check_contact=True)

            # Check if contact with objects
            if check_collision_config and self.gripper_contact(**check_collision_config):
                return False, p.getLinkState(self.robot_id, 7)[0:2]

            # Check xyz and rpy error
            real_xyz, real_xyzw = p.getLinkState(
                self.robot_id, 7)[0:2]
            roll, pitch, yaw = p.getEulerFromQuaternion(orn)
            real_roll, real_pitch, real_yaw = p.getEulerFromQuaternion(real_xyzw)
            if np.linalg.norm(np.array((x, y, z)) - real_xyz) < 0.001 \
                    and np.abs((roll - real_roll, pitch - real_pitch, yaw - real_yaw)).sum() < 0.001:
                if verbose:
                    print('Reach target with', _, 'steps')
                return True, (real_xyz, real_xyzw)

        return False, p.getLinkState(self.robot_id, 7)[0:2]

    def calc_z_offset(self, gripper_opening_length: float):
        gripper_opening_length = np.clip(
            gripper_opening_length, *self.gripper_open_limit)
        gripper_opening_angle = 0.715 - \
            math.asin((gripper_opening_length - 0.010) / 0.1143)
        # if self.gripper_type == '140':
        gripper_length = 10.3613 * \
            np.sin(1.64534-0.24074 * (gripper_opening_angle / np.pi)) - 10.1219
        # else:
        #     gripper_length = 1.231 - 1.1
        return gripper_length

    def auto_close_gripper(self, step: int = 120, check_contact: bool = False) -> bool:
        # Get initial gripper open position
        initial_position = p.getJointState(self.robot_id, self.joints[self.mimicParentName].id)[0]
        initial_position = math.sin(0.715 - initial_position) * 0.1143 + 0.010
        for step_idx in range(1, step):
            current_target_open_length = initial_position - step_idx / step * initial_position

            self.move_gripper(current_target_open_length, 1)
            if current_target_open_length < 1e-5:
                return False

            if check_contact and self.gripper_contact():
                return True
        return False

    def check_target_reached(self, obj_id):
        aabb = p.getAABB(self.target_id, -1)
        x_min, x_max = aabb[0][0], aabb[1][0]
        y_min, y_max = aabb[0][1], aabb[1][1]
        pos = p.getBasePositionAndOrientation(obj_id)
        x, y = pos[0][0], pos[0][1]
        if x > x_min and x < x_max and y > y_min and y < y_max:
            return True
        return False
    
    def grasp(self, pos: tuple, roll: float, gripper_opening_length: float, obj_height: float, obj_name: str, debug: bool = False):
        """
        Method to perform grasp
        pos [x y z]: The axis in real-world coordinate
        roll: float,   for grasp, it should be in [-pi/2, pi/2)
        """

        obj_height = obj_height
        gripper_opening_length = gripper_opening_length


        x, y, z = pos
        # Substracht gripper finger length from z
        z -= 0.06
        z = np.clip(z, *self.ee_position_limit[2])

        # Move above target
        self.move_gripper(0.1)
        orn = p.getQuaternionFromEuler([roll, np.pi/2, 0.0])
        GRIPPER_MOVING_HEIGHT = 1.25
        self.move_ee([x, y, GRIPPER_MOVING_HEIGHT, orn])

        # Reduce grip to get a tighter grip
        gripper_opening_length *= 0.60

        # Grasp and lift object
        z_offset = self.calc_z_offset(gripper_opening_length)
        self.move_ee([x, y, z , orn]) # + z_offset
        self.auto_close_gripper(check_contact=True)
        for _ in range(40):
            self.step_simulation()
        self.move_ee([x, y, GRIPPER_MOVING_HEIGHT, orn])

        return x, y, GRIPPER_MOVING_HEIGHT, orn
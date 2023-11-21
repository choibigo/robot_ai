from grasping.grasp_generator import GraspGenerator
from grasping.environment.utilities import Camera
import numpy as np
import pybullet as p
import argparse
import os
import sys
import cv2
import math
import matplotlib.pyplot as plt
import time
import clip
import glob
import torch
import torchvision.transforms
from PIL import Image
from skeleton.detector import SkeletonDetection 

class GrasppingScenarios():

    def __init__(self):

        self.IMG_SIZE = 224
        self.network_path = 'grasping/trained_models/GR_ConvNet/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98'
        sys.path.append('grasping/trained_models/GR_ConvNet')
        
        self.depth_radius = 1
        self.ATTEMPTS = 3
    
    def show(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Tabletop view', image)
        cv2.waitKey(5)
                
    def draw_predicted_grasp(self,grasps,color = [0,0,1],lineIDs = []):
        x, y, z, yaw, opening_len, obj_height = grasps

        gripper_size = opening_len + 0.02 
        finger_size = 0.075
        # lineIDs = []
        lineIDs.append(p.addUserDebugLine([x, y, z], [x, y, z+0.15],color, lineWidth=6))

        lineIDs.append(p.addUserDebugLine([x - gripper_size*math.sin(yaw), y - gripper_size*math.cos(yaw), z], 
                                    [x + gripper_size*math.sin(yaw), y + gripper_size*math.cos(yaw), z], 
                                    color, lineWidth=6))

        lineIDs.append(p.addUserDebugLine([x - gripper_size*math.sin(yaw), y - gripper_size*math.cos(yaw), z], 
                                    [x - gripper_size*math.sin(yaw), y - gripper_size*math.cos(yaw), z-finger_size], 
                                    color, lineWidth=6))
        lineIDs.append(p.addUserDebugLine([x + gripper_size*math.sin(yaw), y + gripper_size*math.cos(yaw), z], 
                                    [x + gripper_size*math.sin(yaw), y + gripper_size*math.cos(yaw), z-finger_size], 
                                    color, lineWidth=6))
        
        return lineIDs
    
    def remove_drawing(self,lineIDs):
        for line in lineIDs:
            p.removeUserDebugItem(line)
    
    def dummy_simulation_steps(self,n):
        for _ in range(n):
            p.stepSimulation()

    def capture_and_save_image(self, basePosition, object_name, output_path):

        camera_position = [basePosition[0], basePosition[1], basePosition[2] + 0.5]
        target_position = basePosition
        up_vector = [1, 0, 0]

        view_matrix = p.computeViewMatrix(camera_position, target_position, up_vector)
        projection_matrix = p.computeProjectionMatrixFOV(60, 1.0, 0.02, 10.0)

        width, height, rgb_image, _, _ = p.getCameraImage(width=224, height=224, viewMatrix=view_matrix, projectionMatrix=projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgba = np.reshape(rgb_image, (height, width, 4))

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
        rgb_image = __rgba2rgb(rgba)
        image = Image.fromarray(rgb_image)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        image_path = os.path.join(output_path, f"captured_image_{object_name}.png")
        image.save(image_path)

    def scenario(self, env, text_query, goal_point, device='cpu', vis=True, output_path='grasping/results/images'):
        rgb_image, _ = env.camera_set(env.camera_2_config)
        self.show(rgb_image)
        
        object_positions = [
            [-0.4, 0.18, 0.695],    # YcbBanana
            [-0.8, -0.18, 0.71],    # YcbMustardBottle
            [-0.4, -0.18, 0.455],   # YcbPottedMeatCan
            [-0.8, 0.18, 0.53]      # YcbScissors
        ]
        object_names = ["YcbBanana", "YcbMustardBottle", "YcbPottedMeatCan", "YcbScissors"]

        for basePosition, obj_name in zip(object_positions, object_names):
            try:
                self.capture_and_save_image(basePosition, obj_name, output_path)
            except Exception as e:
                print(f"An error occurred with {obj_name}: {e}")
                continue

        clip_model, preprocess = clip.load('ViT-B/16', device=device, jit=False)
        clip_model.eval().float()

        path = f'grasping/results/images/*.png'
        images = []
        for filename in sorted(glob.glob(path)):
            print(filename)
            image = Image.open(filename).convert('RGB')
            images.append(image)

        image_input = torch.stack([preprocess(im) for im in images]).to(device)
        text_query = text_query
        text_input = clip.tokenize([text_query]).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_input)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = image_features @ text_features.T

        similarity_np = similarity.cpu().numpy()
        image_files = sorted(np.array(glob.glob(path)))

        for i in range(similarity_np.shape[0]):
            print(f'Score: {float(similarity_np[i]):.8f}, Image file: {image_files[i]}')
        
        max_index = np.argmax(similarity_np)
        print('The most similar images: ', image_files[max_index])

        plt.figure(figsize=(7, 7))
        image = Image.open(image_files[max_index])
        tensor_image = torchvision.transforms.ToTensor()(image)
        plt.imshow(tensor_image.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.title(f'{text_query}')
        if not os.path.exists('grasping/results/images/clip_result'):
            os.makedirs('grasping/results/images/clip_result')
        plt.savefig(f'grasping/results/images/clip_result/clip_result.png', bbox_inches='tight', pad_inches=1)
        plt.show(block=False)
        plt.pause(5)
        plt.close()

        object_position = object_positions[max_index]
        object_name = object_names[max_index]

        fig = plt.figure(figsize=(10, 10))
        camera = Camera((object_position[0], object_position[1], object_position[2] + 1.0), (object_position[0], object_position[1], object_position[2]), 0.2, 2.0, (self.IMG_SIZE, self.IMG_SIZE), 40)
        generator = GraspGenerator(self.network_path, camera, self.depth_radius, fig, self.IMG_SIZE,'GR_ConvNet', device)
        
        self.dummy_simulation_steps(50)
             
        try: 
            idx = 0

            rgb, depth, _ = camera.get_cam_img()
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                
            grasps, save_name = generator.predict_grasp(rgb, depth, n_grasps=3, show_output=True)
            if (grasps == []):
                self.dummy_simulation_steps(30)
                print ("could not find a grasp point!")
                
            if vis:
                LID =[]
                for g in grasps:
                    LID = self.draw_predicted_grasp(g,color=[1,0,1],lineIDs=LID)
                time.sleep(0.5)
                self.remove_drawing(LID)
                self.dummy_simulation_steps(10)
                    
            if (idx > len(grasps)-1):  
                print ("idx = ", idx)
                if len(grasps) > 0 :
                    idx = len(grasps)-1

            lineIDs = self.draw_predicted_grasp(grasps[idx])
                
            # perform object grasping and manipulation
            x, y, z, yaw, opening_len, obj_height = grasps[idx]
            x, y, z, orn = env.grasp((x, y, z), yaw, opening_len, obj_height, object_name)

            skeleton_detector = SkeletonDetection()
            for _ in range(30):
                rgb_image, _ = env.camera_set(env.camera_1_config)
                detection_image, skeleton_info = skeleton_detector.detection(rgb_image)
                real_coordinate_from_cramera_image = env.get_point_cloud()
                skeleton_detector.show(detection_image)
                env.step_simulation()
        
            
        except Exception as ex:
            print("An exception occurred during the experiment!!!")
            print(f"Exception: {ex}")
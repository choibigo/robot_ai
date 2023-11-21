import cv2
import mediapipe as mp

LANDMARKER = {
0 : "nose",
1 : "left eye (inner)",
2 : "left eye",
3 : "left eye (outer)",
4 : "right eye (inner)",
5 : "right eye",
6 : "right eye (outer)",
7 : "left ear",
8 : "right ear",
9 : "mouth (left)",
10 : "mouth (right)",
11 : "left shoulder",
12 : "right shoulder",
13 : "left elbow",
14 : "right elbow",
15 : "left wrist",
16 : "right wrist",
17 : "left pinky",
18 : "right pinky",
19 : "left index",
20 : "right index",
21 : "left thumb",
22 : "right thumb",
23 : "left hip",
24 : "right hip",
25 : "left knee",
26 : "right knee",
27 : "left ankle",
28 : "right ankle",
29 : "left heel",
30 : "right heel",
31 : "left foot index",
32 : "right foot index"}

class SkeletonDetection:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.max_visibility = 0
        self.skeleton_info = []
        self.skeleton_image = None

    def detection(self, image):
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        
        sum_visibility = 0
        if results.pose_landmarks is not None:
            (h, w) = image.shape[:2]
            temp_skeleton_info = []
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                joint_info = {
                    'x' : int(landmark.x*w),
                    'y' : int(landmark.y*h),
                    'name' : LANDMARKER[idx]
                }
                sum_visibility += landmark.visibility
                temp_skeleton_info.append(joint_info)

        if self.skeleton_image is None:
            self.skeleton_image = image

        if self.max_visibility < sum_visibility:
            self.max_visibility = sum_visibility
            self.skeleton_info = temp_skeleton_info
            self.skeleton_image = image
            
        return self.skeleton_image, self.skeleton_info
    
    def show(self, image):
        cv2.imshow('Skeleton Detection', image)
        cv2.waitKey(5)
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
        
        skeleton_info = []
        if results.pose_landmarks is not None:
            (h, w) = image.shape[:2]
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                temp = {
                    'x' : int(landmark.x*w),
                    'y' : int(landmark.y*h),
                    'name' : LANDMARKER[idx]
                }
                skeleton_info.append(temp)

        return image, skeleton_info
    
    def show(self, image):
        cv2.imshow('Skeleton Detection', image)
        cv2.waitKey(5)

if __name__ == "__main__":
    
    import copy

    person_image = cv2.imread(r'/workspace/test_space/daewon/person_image/person3.jpg', 1)
    (h, w) = person_image.shape[:2]

    detection_instance = SkeletonDetection()
    skeleton_image, skeleton_info = detection_instance.detection(person_image)

    show_image  = copy.deepcopy(skeleton_image)
    for point_info in skeleton_info:
        show_image = cv2.putText(show_image, point_info['name'], (point_info['x'], point_info['y']), 1, 1, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow('MediaPipe Pose', show_image)
    cv2.waitKey(0)
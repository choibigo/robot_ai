import cv2
import mediapipe as mp

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
        return image, results
    
    def show(self, image):
        cv2.imshow('MediaPipe Pose', image)
        cv2.waitKey(5)

if __name__ == "__main__":
    
    person_image = cv2.imread(r'/workspace/test_space/daewon/person_image/person3.jpg', 1)
    
    detection_instance = SkeletonDetection()
    skeleton_image = detection_instance.detection(person_image)
    cv2.imshow('MediaPipe Pose', skeleton_image)
    cv2.waitKey(0)
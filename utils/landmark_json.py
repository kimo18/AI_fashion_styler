import cv2
import mediapipe as mp
import json
import os
media_pipe_70_indices = [
    127, 234, 93, 132, 58, 172, 136, 150, 152, 377,
    400, 436, 418, 394, 362, 263, 46, 53, 65, 55,
    107, 6, 195, 5, 4, 1, 19, 94, 2, 164,
    3, 97, 263, 362, 385, 374, 373, 390, 249, 133,
    173, 157, 154, 155, 145, 153, 78, 191, 80, 308,
    324, 317, 13, 82, 312, 14, 87, 317, 78, 308,
    234, 454, 127, 454, 234, 454, 152, 377, 152, 400
]

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
# Read the image
# annotated_image = image.copy()

# Prepare final result structure
output = {
    "version": 1.3,
    "people": [{
        "person_id": [-1],
        "pose_keypoints_2d": [],
        "face_keypoints_2d": [],
        "hand_left_keypoints_2d": [],
        "hand_right_keypoints_2d": [],
        "pose_keypoints_3d":[],
        "face_keypoints_3d":[],
        "hand_left_keypoints_3d":[],
        "hand_right_keypoints_3d":[]
    }]
}


class Landmark_Json():
    def __init__(self,image_dir,image_name,landmark_json_dir):
        self.landmark_json_dir = landmark_json_dir
        jpg_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.image = cv2.imread(os.path.join(image_dir, jpg_files[0]))
        self.image_name = image_name
        with mp_holistic.Holistic(static_image_mode=True) as holistic:
            self.results = holistic.process(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))

            



    # Helper to flatten and append keypoints
    def extract_keypoints(self,landmarks, image_shape):
        h, w, _ = image_shape
        keypoints = []
        for lm in landmarks:
            x = lm.x * w
            y = lm.y * h
            c = lm.visibility if hasattr(lm, 'visibility') else lm.z  # confidence
            keypoints.extend([x, y, c])
        return keypoints
    
    def get_landmarks_json(self):
        if self.results.face_landmarks:
            # mp_drawing.draw_landmarks(annotated_image, self.results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
            h, w, _ = self.image.shape
            face_keypoints_2d = []
            
            for i in media_pipe_70_indices:
                if i < len(self.results.face_landmarks.landmark):
                    lm = self.results.face_landmarks.landmark[i]
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    # MediaPipe face landmarks don't have 'visibility', so using 'z' as proxy confidence
                    c = lm.z if hasattr(lm, 'z') else 0.0
                    face_keypoints_2d.extend([x, y, c])
                else:
                    face_keypoints_2d.extend([0, 0, 0])
            
            output["people"][0]["face_keypoints_2d"] = face_keypoints_2d

        if self.results.pose_landmarks:
            # mp_drawing.draw_landmarks(annotated_image, self.results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            pose_keypoints = self.extract_keypoints(self.results.pose_landmarks.landmark, self.image.shape)
            output["people"][0]["pose_keypoints_2d"] = pose_keypoints

        if self.results.left_hand_landmarks:
            # mp_drawing.draw_landmarks(annotated_image, self.results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            left_hand_keypoints = self.extract_keypoints(self.results.left_hand_landmarks.landmark, self.image.shape)
            output["people"][0]["hand_left_keypoints_2d"] = left_hand_keypoints

        if self.results.right_hand_landmarks:
            # mp_drawing.draw_landmarks(annotated_image, self.results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            right_hand_keypoints = self.extract_keypoints(self.results.right_hand_landmarks.landmark, self.image.shape)
            output["people"][0]["hand_right_keypoints_2d"] = right_hand_keypoints
        
        with open(os.path.join(self.landmark_json_dir,f'{self.image_name}_keypoints.json'), "w") as json_file:
            json.dump(output, json_file, indent=4)

    # Run holistic model in static mode
    
import cv2
import mediapipe as mp
import numpy as np
import math
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Video Feed
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 960)

prev_frame_time = 0
new_frame_time = 0

# Setup MediaPipe Instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        success, frame = cap.read()

        # Recolor Image (mediapipe expects RGB but cv2 gives BGR)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False # Saves memory

        # Make Detection
        results = pose.process(image)
        
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract Landmarks
        try: # Might not extract certain landmarks
            landmarks = results.pose_landmarks.landmark

            # Coordinates for specific joints
            leftShoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
            leftElbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
            leftWrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]

            # Calculate Angle
            leftAngle = int(calculate_angle(leftShoulder, leftElbow, leftWrist))

            # Display Angle
            cv2.putText(image, str(leftAngle), tuple(np.multiply((leftElbow[0], leftElbow[1]), [1280 + 500, 960 + 400]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 2, cv2.LINE_AA)
        except:
            pass

        def calculate_angle(a, b, c):
            A = np.array(a) # First
            B = np.array(b) # Mid
            C = np.array(c) # End

            BA = np.subtract(B, A)
            BC = np.subtract(B, C)

            # Calculate the dot product of BA and BC
            dot_product = np.dot(BA, BC)

            # Calculate the magnitudes of BA and BC
            magnitude_BA = np.linalg.norm(BA)
            magnitude_BC = np.linalg.norm(BC)

            # Calculate the cosine of the angle between BA and BC
            cosine_angle = dot_product / (magnitude_BA * magnitude_BC)

            # Calculate the angle in radians
            angle_radians = np.arccos(cosine_angle)

            # Convert the angle from radians to degrees if necessary
            angle_degrees = np.degrees(angle_radians)

            return angle_degrees

        # Draw Landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), # Joint (color is in BGR format)
        mp_drawing.DrawingSpec(color=(76,71,255), thickness=2, circle_radius=2) # Connection
        )

        # mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        
        cv2.imshow("Image", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
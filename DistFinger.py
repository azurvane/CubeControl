import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands and Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

left_finger = mp_hands.HandLandmark.INDEX_FINGER_TIP
right_finger = mp_hands.HandLandmark.INDEX_FINGER_TIP
IMPORTANT_POINTS = [
    mp_hands.HandLandmark.INDEX_FINGER_TIP,
    mp_hands.HandLandmark.THUMB_TIP,
]

def draw_important_point(frame, hand_landmarks):
    h, w, _ = frame.shape
    for idx in IMPORTANT_POINTS:
        lm = hand_landmarks.landmark[idx]
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 48, (255, 253, 208), -1)

def draw_hand_landmark(frame, hand_landmarks, hand_label):
    h, w, _ = frame.shape
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    x, y = int(wrist.x * w), int(wrist.y * h)
    cv2.putText(frame, hand_label, (x - 30, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def cal_draw_distance(frame, left_index, right_index):
    dx = left_index[0] - right_index[0]
    dy = left_index[1] - right_index[1]
    distance = np.linalg.norm((dx,dy))
    text = f"distnacne: {distance:.3f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return distance

def run_hand_tracking():
    
    # Initialize video capture
    cam = cv2.VideoCapture(index=1)
    
    # Initialize MediaPipe Hands
    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        
        # Frame capture loop
        while cam.isOpened():
            
            # Read a frame from the camera
            success, frame = cam.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            
            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Convert the BGR image to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            left_index = None
            right_index = None
            
            if results.multi_hand_landmarks and results.multi_handedness:                
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                    ):
                    
                    # Get hand label
                    hand_label = handedness.classification[0].label
                    
                    if hand_label.lower() == "left":
                        lm = hand_landmarks.landmark[left_finger]
                        left_index = (lm.x, lm.y)
                    else:
                        lm = hand_landmarks.landmark[right_finger]
                        right_index = (lm.x, lm.y)
                    
                    # Draw circles on important points
                    draw_important_point(frame, hand_landmarks)
                    
                    # Draw hand landmarks
                    draw_hand_landmark(frame, hand_landmarks, hand_label)
                
                if left_index and right_index:
                    distance = cal_draw_distance(frame, left_index, right_index)
            
            # display the frame
            cv2.imshow('Hand Tracking', frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cam.release()


if __name__ == "__main__":
    run_hand_tracking()


import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles


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
            
            # Convert the BGR image to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    IMPORTANT_POINTS = [
                        mp_hands.HandLandmark.INDEX_FINGER_TIP,
                        mp_hands.HandLandmark.THUMB_TIP,
                    ]
                    
                    h, w, _ = frame.shape
                    for idx in IMPORTANT_POINTS:
                        lm = hand_landmarks.landmark[idx]
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 48, (255, 253, 208), -1)
            
            # display the frame
            cv2.imshow('Hand Tracking', cv2.flip(frame, 1))
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cam.release()


if __name__ == "__main__":
    run_hand_tracking()


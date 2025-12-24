import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands and Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles


index_tip = mp_hands.HandLandmark.INDEX_FINGER_TIP 
index_mcp = mp_hands.HandLandmark.INDEX_FINGER_MCP
thumb_tip = mp_hands.HandLandmark.THUMB_TIP 
thump_mcp = mp_hands.HandLandmark.THUMB_MCP
wrist = mp_hands.HandLandmark.WRIST
pinky_mcp = mp_hands.HandLandmark.PINKY_MCP



'''
~0.0 → ~90° (fingers perpendicular)

~0.5 → ~60°

~0.7 → ~45°

~0.94 → ~20°

1.0 → 0° (fully aligned)
'''
def palm_plane(hand_landmarks, hand_label):
    w = hand_landmarks.landmark[wrist]
    i = hand_landmarks.landmark[index_mcp]
    p = hand_landmarks.landmark[pinky_mcp]
    
    a = np.array([w.x - i.x, w.y - i.y, w.z - i.z])
    b = np.array([w.x - p.x, w.y - p.y, w.z - p.z])
    
    z_axis = np.cross(a, b)
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    if hand_label == "Right":
        # Right hand convention: check against index direction
        index_dir = np.array([i.x - w.x, i.y - w.y, i.z - w.z])
        if np.dot(z_axis, index_dir) < 0:
            z_axis = -z_axis
    else:  # Left hand
        index_dir = np.array([i.x - w.x, i.y - w.y, i.z - w.z])
        if np.dot(z_axis, index_dir) > 0:  # Opposite for left
            z_axis = -z_axis
    
    x_axis_raw = np.array([i.x - p.x, i.y - p.y, i.z - p.z])
    x_axis = x_axis_raw - np.dot(x_axis_raw, z_axis)*z_axis
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    return np.array([x_axis, y_axis, z_axis])

def angle_fingers(hand_landmarks, hand_label):
    
    i_tip = hand_landmarks.landmark[index_tip]
    i_mcp = hand_landmarks.landmark[index_mcp]
    t_tip = hand_landmarks.landmark[thumb_tip]
    t_mcp = hand_landmarks.landmark[thump_mcp]
    
    vector_index = np.array([i_tip.x - i_mcp.x, i_tip.y - i_mcp.y, i_tip.z - i_mcp.z])
    vector_thumb = np.array([t_tip.x - t_mcp.x, t_tip.y - t_mcp.y, t_tip.z - t_mcp.z])
    
    vector_index = vector_index / np.linalg.norm(vector_index)
    vector_thumb = vector_thumb / np.linalg.norm(vector_thumb)
    
    plam = palm_plane(hand_landmarks, hand_label)
    
    index_projected = vector_index - np.dot(vector_index, plam[2])*plam[2]
    thumb_projected = vector_thumb - np.dot(vector_thumb, plam[2])*plam[2]
    
    index_2d = np.array([np.dot(index_projected, plam[0]), np.dot(index_projected, plam[1])])
    thumb_2d = np.array([np.dot(thumb_projected, plam[0]), np.dot(thumb_projected, plam[1])])
    
    index_2d = index_2d / np.linalg.norm(index_2d)
    thumb_2d = thumb_2d / np.linalg.norm(thumb_2d)
    
    cos_angle = np.dot(index_2d, thumb_2d)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Prevent numerical errors
    angle_rad = np.arccos(cos_angle)
    angle_deg = angle_rad * 180.0 / np.pi
    
    return angle_deg



def hand_angle_tracking():
    
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
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
                    )
                    hand_label = results.multi_handedness[idx].classification[0].label
                    print(angle_fingers(hand_landmarks, hand_label))
            
            # display the frame
            cv2.imshow('Hand Tracking', cv2.flip(frame, 1))
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cam.release()


if __name__ == "__main__":
    hand_angle_tracking()



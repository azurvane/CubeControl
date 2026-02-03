from ursina import *
import cv2
import mediapipe as mp
import threading
from PIL import Image
import numpy as np

# --- MEDIAPIPE SETUP ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2, 
    model_complexity=0,
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

# Global data
hand_data = {"Left": None, "Right": None}
latest_frame = None
frame_lock = threading.Lock()

def get_hand_tracking():
    global hand_data, latest_frame
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: 
            continue
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Store frame
        with frame_lock:
            latest_frame = rgb_frame.copy()
        
        # Process hand tracking
        results = hands.process(rgb_frame)
        temp_data = {"Left": None, "Right": None}
        
        if results.multi_hand_landmarks:
            for idx, hand_handedness in enumerate(results.multi_handedness):
                label = hand_handedness.classification[0].label
                temp_data[label] = results.multi_hand_landmarks[idx]
        
        hand_data = temp_data
    
    cap.release()

# Start tracking thread
thread = threading.Thread(target=get_hand_tracking, daemon=True)
thread.start()

# --- URSINA ENGINE SETUP ---
app = Ursina(borderless=False)

# SOLUTION: Use a Sprite instead of Entity for reliable texture updates
from ursina import Sprite

# Create background sprite - Sprites handle dynamic textures better
video_bg = Sprite(
    name='video_background',
    position=window.top_left,
    origin=(-.5, .5),
    scale=(2 * camera.aspect_ratio, 2),
    z=1
)

# Voxel Container
voxel_world = Entity()
placed_voxels = {}
cursor = Entity(model='cube', color=color.cyan, scale=0.5, collider='box')

def update():
    # Update video background
    if latest_frame is not None:
        try:
            with frame_lock:
                # Create PIL Image
                pil_img = Image.fromarray(latest_frame)
                
                # Save to temporary file and load as texture
                temp_path = 'temp_frame.png'
                pil_img.save(temp_path)
                video_bg.texture = temp_path
                
        except Exception as e:
            pass
    
    L = hand_data["Left"]
    R = hand_data["Right"]
    
    # FEATURE: GLOBAL ROTATION (2 HANDS)
    if L and R:
        voxel_world.rotation_y += (R.landmark[9].x - L.landmark[9].x - 0.5) * 50 * time.dt
        voxel_world.rotation_x += (R.landmark[9].y - L.landmark[9].y) * 50 * time.dt
        cursor.visible = False
    else:
        cursor.visible = True
    
    # FEATURE: 3D BUILDING (RIGHT HAND)
    if R:
        index_tip = R.landmark[8]
        thumb_tip = R.landmark[4]
        
        # Mapping to 3D Space
        target_pos = Vec3((index_tip.x - 0.5) * 16, (0.5 - index_tip.y) * 9, 0)
        cursor.position = Vec3(round(target_pos.x), round(target_pos.y), round(target_pos.z))
        
        # Check for Pinch
        dist = Vec2(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y).length()
        if dist < 0.05:
            pos_key = (cursor.x, cursor.y, cursor.z)
            if pos_key not in placed_voxels:
                new_v = Entity(
                    parent=voxel_world, 
                    model='cube', 
                    position=cursor.position, 
                    color=color.cyan, 
                    scale=0.9, 
                    collider='box'
                )
                placed_voxels[pos_key] = new_v

def input(key):
    if key == 'c':
        for v in list(placed_voxels.values()):
            destroy(v)
        placed_voxels.clear()

app.run()
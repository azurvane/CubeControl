from ursina import *
import cv2
import mediapipe as mp
import threading
from PIL import Image

# --- MEDIAPIPE SETUP ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2, 
    model_complexity=0, # Mandatory for smooth M3 performance
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

# Global data
hand_data = {"Left": None, "Right": None}
latest_frame = None

def get_hand_tracking():
    global hand_data, latest_frame
    cap = cv2.VideoCapture(0)
    
    # Set lower resolution to ensure high FPS on the shared thread
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        # Store frame immediately
        latest_frame = frame
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

# Video Background - Placed in the UI layer so it's always behind
video_bg = Entity(parent=camera.ui, model='quad', scale=(1.8, 1), z=10)

# Voxel Container
voxel_world = Entity()
placed_voxels = {}
cursor = Entity(model='cube', color=color.cyan, scale=0.5, mode='wireframe')

def update():
    # 1. Update Video Background
    if latest_frame is not None:
        try:
            # Convert to RGB and then to PIL Image
            img = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            
            # Update the texture directly
            video_bg.texture = Texture(pil_img)
        except Exception as e:
            pass # Prevent crash if a frame is partially corrupted during thread swap

    L = hand_data["Left"]
    R = hand_data["Right"]

    # 2. FEATURE: GLOBAL ROTATION (2 HANDS)
    if L and R:
        # We use a smaller multiplier (50) for smoother control on Mac
        voxel_world.rotation_y += (R.landmark[9].x - L.landmark[9].x - 0.5) * 50 * time.dt
        voxel_world.rotation_x += (R.landmark[9].y - L.landmark[9].y) * 50 * time.dt
        cursor.visible = False
    else:
        cursor.visible = True

    # 3. FEATURE: 3D BUILDING (RIGHT HAND)
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
                new_v = Entity(parent=voxel_world, model='cube', position=cursor.position, 
                               color=color.cyan, scale=0.9, mode='wireframe')
                placed_voxels[pos_key] = new_v

def input(key):
    if key == 'c':
        for v in placed_voxels.values():
            destroy(v)
        placed_voxels.clear()

app.run()
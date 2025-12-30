from ursina import *
import cv2
import numpy as np
from PIL import Image
from panda3d.core import Texture as PandaTexture

app = Ursina(title="AR Cube - Milestone 1", borderless=False)

# macOS UI Fixes
window.fps_counter.enabled = True
window.update_aspect_ratio()

# --- TODO 1: Camera Index Check ---
# Try 0 if 1 is giving a black screen. 
# Index 0 is usually the built-in FaceTime HD camera.
cap = cv2.VideoCapture(0) 

# --- Setup ---
blank_array = np.zeros((480, 640, 3), dtype=np.uint8) 
pil_image = Image.fromarray(blank_array)
video_texture = Texture(pil_image)

video_texture._texture.setup_2d_texture(
    640, 480, 
    PandaTexture.T_unsigned_byte, 
    PandaTexture.F_rgb
)
video_texture.filtering = None

background_screen = Entity(
    model='quad', 
    texture=video_texture, 
    scale=(camera.aspect_ratio * 10, 10), 
    z=20 # Push it back a bit further
)

def update():
    ret, frame = cap.read()
    
    # --- TODO 2: Debugging the "Black Screen" ---
    if not ret or frame is None:
        return
        
    # Validation: If the frame is all zeros, OpenCV isn't "seeing" anything
    if np.all(frame == 0):
        # Hint: This usually means the camera is on but the shutter is closed 
        # or another app is using the camera.
        return

    # Process
    frame = cv2.resize(frame, (640, 480))
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Memory Alignment
    frame_data = np.ascontiguousarray(rgb_frame).tobytes()
    
    # Injection
    video_texture._texture.set_ram_image(frame_data)
    video_texture.apply()

def input(key):
    if key == 'escape':
        cap.release()
        quit()

app.run()
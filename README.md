# Hand Gesture Controlled 3D Cube

This project implements a real-time system where a 3D wireframe cube is rendered over a live camera feed and controlled using hand gestures. The cube’s rotation and scale are manipulated using finger positions, and only the user’s hands are visible against a black background.

---

## Project Tasks and Libraries Used

### 1. Camera Input

**Task:** Capture real-time video frames from the webcam.
**Library Used:** OpenCV

---

### 2. Hand Landmark Detection

**Task:** Detect hands and extract finger joint landmarks from each video frame.
**Library Used:** MediaPipe Hands

---

### 3. Hand Extraction and Background Masking

**Task:** Isolate hands from the video feed and replace the background with black.
**Libraries Used:** MediaPipe Hands, OpenCV

---

### 4. Gesture Detection

**Task:** Analyze hand landmarks to detect predefined gestures based on finger positions and distances.
**Library Used:** NumPy

---

### 5. Gesture State Management

**Task:** Maintain stable gesture states and control when gesture-based actions start and stop.
**Library Used:** Custom Gesture State Machine (logic built using NumPy)

---

### 6. Gesture-to-Parameter Mapping

**Task:** Convert detected gestures into cube parameters such as rotation and scale.
**Library Used:** NumPy

---

### 7. 3D Rendering

**Task:** Render a 3D wireframe cube with glowing edges using GPU acceleration.
**Library Used:** ModernGL

---

### 8. OpenGL Context and Window Management

**Task:** Create and manage the OpenGL rendering window and rendering loop.
**Library Used:** GLFW or pyglet

---

### 9. Video and 3D Compositing

**Task:** Combine the processed video frame with the rendered 3D cube into a single output.
**Libraries Used:** ModernGL, OpenCV

---

## License

Licensed under the MIT License.

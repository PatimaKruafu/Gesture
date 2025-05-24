import mediapipe as mp
import cv2
import numpy as np
import time # For timestamp
# You'll need this for converting landmarks to the format drawing_utils expects
from mediapipe.framework.formats import landmark_pb2

# --- Global variable to store the latest frame with landmarks ---
# This is a simple way to pass the annotated image from the callback to the main thread.
# For more robust applications, consider using a thread-safe queue.
latest_annotated_frame = None

hand_landmarker_path = 'tasks/hand_landmarker.task' # Make sure this path is correct

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands # For HAND_CONNECTIONS

# Create a hand landmarker instance with the live stream mode:
def result_callback_for_drawing(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_annotated_frame
    # print(f'Hand landmarker result: {timestamp_ms}') # Less verbose than printing the whole result

    annotated_image_np = output_image.numpy_view().copy() # Convert mp.Image to mutable NumPy array (RGB)

    if result.hand_landmarks:
        for hand_landmarks_list in result.hand_landmarks: # This is a list of landmarks for a single hand
            # Create a NormalizedLandmarkList protobuf object
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                for landmark in hand_landmarks_list
            ])

            # Draw the landmarks
            mp_drawing.draw_landmarks(
                annotated_image_np, # Image to draw on (RGB)
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4), # Landmark style
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)  # Connection style
            )
    
    # Convert RGB back to BGR for OpenCV display
    annotated_image_bgr = cv2.cvtColor(annotated_image_np, cv2.COLOR_RGB2BGR)
    latest_annotated_frame = annotated_image_bgr


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_landmarker_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2, # Max number of hands to detect
    result_callback=result_callback_for_drawing) # Use our new callback

# Main loop
with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(1) # Try 0 if 1 doesn't work, or your specific camera index
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    cv2.namedWindow('MediaPipe Hand Landmarks', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('MediaPipe Hand Landmarks', 800, 600) # Optional resizing

    frame_timestamp_ms = 0

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            break

        frame_timestamp_ms = int(time.time() * 1000) # Get current time in milliseconds

        # Convert the BGR frame to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Create a MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Perform hand landmarking asynchronously
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        # Display the frame:
        # If latest_annotated_frame is available, show it. Otherwise, show the original.
        display_frame = frame_bgr # Default to original frame
        if latest_annotated_frame is not None:
            display_frame = latest_annotated_frame
        
        cv2.imshow('MediaPipe Hand Landmarks', display_frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

print("Program finished.")
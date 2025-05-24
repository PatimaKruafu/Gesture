import mediapipe as mp
import cv2
import numpy as np
import time
from mediapipe.framework.formats import landmark_pb2

from mediapipe.tasks.python.components.processors import classifier_options


# --- Global variables ---
latest_annotated_frame = None
latest_gestures_data = [] # Will store (handedness, gesture_name) tuples

# --- Model Paths ---
hand_landmarker_path = 'tasks/hand_landmarker.task' # Make sure this path is correct
gesture_recognizer_path = 'tasks/gesture_recognizer.task' # Make sure this path is correct

# --- MediaPipe Task Setup ---
BaseOptions = mp.tasks.BaseOptions
# Hand Landmarker
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
# Gesture Recognizer
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands # For HAND_CONNECTIONS

# --- Result Callbacks ---

# Callback for Hand Landmarker
def hand_landmarker_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_annotated_frame
    # print(f'Hand landmarker result at {timestamp_ms}')

    annotated_image_np = output_image.numpy_view().copy() # RGB

    if result.hand_landmarks:
        for hand_landmarks_list in result.hand_landmarks:
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                for landmark in hand_landmarks_list
            ])

            mp_drawing.draw_landmarks(
                annotated_image_np,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
    
    latest_annotated_frame = annotated_image_np # Store as RGB, convert to BGR just before display

# Callback for Gesture Recognizer
def gesture_recognizer_callback(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_gestures_data
    # print(f'Gesture recognizer result at {timestamp_ms}')
    
    SCORE_THRESHOLD = 0.6 # Define your score threshold here

    current_gestures = []
    if result.gestures:
        for i, gesture_categories_for_hand in enumerate(result.gestures): # gesture_categories_for_hand is a list of Category objects
            if gesture_categories_for_hand: # If any gestures recognized for this hand
                # Iterate through recognized gestures for this hand and pick the first one above threshold
                for gesture_category in gesture_categories_for_hand:
                    if gesture_category.score >= SCORE_THRESHOLD:
                        handedness = "Unknown Hand"
                        if result.handedness and len(result.handedness) > i and result.handedness[i]:
                            handedness_obj = result.handedness[i][0] # Assuming one handedness category per hand
                            handedness = handedness_obj.display_name if handedness_obj.display_name else handedness_obj.category_name
                        
                        current_gestures.append((handedness, gesture_category.category_name, gesture_category.score))
                        break # Got the highest scoring gesture above threshold for this hand
    latest_gestures_data = current_gestures


# --- Options for Tasks ---
hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_landmarker_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=hand_landmarker_callback)

gesture_options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=gesture_recognizer_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=gesture_recognizer_callback,
    # Use the imported classifier_options here
    # canned_gestures_classifier_options=classifier_options.ClassifierOptions(score_threshold=0.6)
    # If you were using custom gestures, it would be:
    # custom_gestures_classifier_options=classifier_options.ClassifierOptions(score_threshold=0.6)
    )


# --- Main Program ---
def main():
    global latest_annotated_frame, latest_gestures_data

    # Create landmarker and recognizer instances
    with HandLandmarker.create_from_options(hand_options) as landmarker, \
         GestureRecognizer.create_from_options(gesture_options) as recognizer:

        cap = cv2.VideoCapture(0) # Try 0 or 1 or your specific camera index
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        cv2.namedWindow('MediaPipe Hand Landmarks & Gestures', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('MediaPipe Hand Landmarks & Gestures', 1000, 750)

        frame_timestamp_ms = 0

        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                break

            frame_timestamp_ms = int(time.time() * 1000)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Asynchronously process with both tasks
            landmarker.detect_async(mp_image, frame_timestamp_ms)
            recognizer.recognize_async(mp_image, frame_timestamp_ms)

            # --- Displaying the results ---
            display_frame_bgr = frame_bgr.copy() # Start with the original BGR frame

            # If hand landmarker has produced an annotated frame (RGB), use it and convert to BGR
            if latest_annotated_frame is not None:
                display_frame_bgr = cv2.cvtColor(latest_annotated_frame, cv2.COLOR_RGB2BGR)
            
            # Draw gesture information on the (potentially annotated) BGR frame
            if latest_gestures_data:
                y_offset = 30
                for handedness_str, gesture_name, score in latest_gestures_data:
                    text = f"{handedness_str}: {gesture_name} ({score:.2f})"
                    cv2.putText(display_frame_bgr, text, (10, y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA) # Red text
                    y_offset += 30
            
            cv2.imshow('MediaPipe Hand Landmarks & Gestures', display_frame_bgr)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Program finished.")

if __name__ == '__main__':
    main()
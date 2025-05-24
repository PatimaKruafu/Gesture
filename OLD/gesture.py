import mediapipe as mp
import cv2
import numpy as np
import time
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.components.processors import classifier_options as mp_classifier_options




# --- PyAutoGUI for controlling mouse/keyboard ---
import pyautogui
pyautogui.FAILSAFE = False # Disable failsafe for testing (be careful!)

# --- Global variables ---
latest_annotated_frame_rgb = None # RGB frame with landmarks drawn
latest_hand_landmarks_data = []   # List of (handedness, landmarks_list)
latest_gestures_data = []         # List of (handedness, gesture_name, score)

# --- Model Paths ---
hand_landmarker_path = 'tasks/hand_landmarker.task'
gesture_recognizer_path = 'tasks/gesture_recognizer.task'

# --- MediaPipe Task Setup ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# --- Landmark Constants ---
INDEX_FINGER_TIP = 8
THUMB_TIP = 4
MIDDLE_FINGER_TIP = 12

# --- HCI Control States & Thresholds ---
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
PINCH_THRESHOLD = 0.06  # Adjust based on your camera and hand size for clicking
CURSOR_SMOOTHING = 0.3 # 0.0 = no smoothing, 1.0 = no movement. Lower is more responsive.
prev_cursor_x, prev_cursor_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 # Initial smoothed position
is_pinching_for_click = False # State for click action

# --- Result Callbacks ---
def hand_landmarker_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_annotated_frame_rgb, latest_hand_landmarks_data

    annotated_image_np_rgb = output_image.numpy_view().copy() # RGB
    current_hand_data = []

    if result.hand_landmarks:
        for i, hand_landmarks_list in enumerate(result.hand_landmarks):
            # Store raw landmark data
            handedness_str = "Unknown"
            if result.handedness and len(result.handedness) > i and result.handedness[i]:
                handedness_obj = result.handedness[i][0]
                handedness_str = handedness_obj.display_name if handedness_obj.display_name else handedness_obj.category_name
            current_hand_data.append((handedness_str, hand_landmarks_list))

            # Draw landmarks for visualization
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks_list
            ])
            mp_drawing.draw_landmarks(
                annotated_image_np_rgb,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
    
    latest_hand_landmarks_data = current_hand_data
    latest_annotated_frame_rgb = annotated_image_np_rgb

# --- Result Callbacks ---
# ... (hand_landmarker_callback remains the same) ...

def gesture_recognizer_callback(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_gestures_data
    
    GESTURE_SCORE_THRESHOLD = 0.7 # Define your score threshold here

    current_gestures = []
    if result.gestures:
        for i, gesture_categories_for_hand in enumerate(result.gestures): # gesture_categories_for_hand is a list of Category objects
            if gesture_categories_for_hand: # If any gesture categories recognized for this hand
                # Iterate through recognized gestures for this hand and pick the first one above threshold
                for gesture_category in gesture_categories_for_hand:
                    if gesture_category.score >= GESTURE_SCORE_THRESHOLD:
                        handedness_str = "Unknown"
                        if result.handedness and len(result.handedness) > i and result.handedness[i]:
                            handedness_obj = result.handedness[i][0] # Assuming one handedness category per hand
                            handedness_str = handedness_obj.display_name if handedness_obj.display_name else handedness_obj.category_name
                        
                        current_gestures.append((handedness_str, gesture_category.category_name, gesture_category.score))
                        # print(f"Gesture: {handedness_str}, {gesture_category.category_name}, {gesture_category.score:.2f}") # For debugging
                        break # Got the highest scoring gesture above threshold for this hand
    latest_gestures_data = current_gestures

# --- Options for Tasks ---
hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_landmarker_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2, # Detect up to 2 hands
    min_hand_detection_confidence=0.5, # Keep these if your version supports them
    min_hand_presence_confidence=0.5,  # Keep these if your version supports them
    min_tracking_confidence=0.5,       # Keep these if your version supports them
    result_callback=hand_landmarker_callback)

gesture_options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=gesture_recognizer_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    # min_hand_detection_confidence=0.5, # Optional, depending on your MP version
    # min_hand_presence_confidence=0.5,  # Optional, depending on your MP version
    # min_tracking_confidence=0.5,       # Optional, depending on your MP version
    result_callback=gesture_recognizer_callback
    # REMOVE: canned_gestures_classifier_options=classifier_options_module.ClassifierOptions(score_threshold=0.7)
)

# --- Helper function to calculate distance ---
def calculate_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

# --- Main Program ---
def main():
    global latest_annotated_frame_rgb, latest_hand_landmarks_data, latest_gestures_data
    global prev_cursor_x, prev_cursor_y, is_pinching_for_click

    with HandLandmarker.create_from_options(hand_options) as landmarker, \
         GestureRecognizer.create_from_options(gesture_options) as recognizer:

        cap = cv2.VideoCapture(0)

        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        window_name = 'MediaPipe HCI Control'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        frame_timestamp_ms = 0
        last_zoom_time = 0
        ZOOM_COOLDOWN = 0.5 # Seconds

        ######FPS CHECK
        # Before the while loop
        prev_time = 0
        
        ######FPS CHECK


        frame_count = 0
        PROCESS_EVERY_N_FRAMES = 2 # Process every other frame

        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                break
            
            # Flip the frame horizontally for a more intuitive "mirror" effect
            frame_bgr = cv2.flip(frame_bgr, 1)

            frame_timestamp_ms = int(time.time() * 1000)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)


            frame_count += 1
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                # Run landmarker.detect_async and recognizer.recognize_async
                landmarker.detect_async(mp_image, frame_timestamp_ms)
                recognizer.recognize_async(mp_image, frame_timestamp_ms)
                pass

            #landmarker.detect_async(mp_image, frame_timestamp_ms)
            #recognizer.recognize_async(mp_image, frame_timestamp_ms)

            display_frame_bgr = frame_bgr.copy() # Start with the original (flipped) BGR frame
            if latest_annotated_frame_rgb is not None:
                # The annotated frame is already flipped if output_image from callback is used directly
                # and output_image is based on the input mp_image (which was made from flipped frame_rgb)
                display_frame_bgr = cv2.cvtColor(latest_annotated_frame_rgb, cv2.COLOR_RGB2BGR)


            ###
            # Inside the while loop, after processing
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(display_frame_bgr, f"FPS: {fps:.2f}", (display_frame_bgr.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            ###


            # --- HCI Logic ---
            active_hand_landmarks = None
            active_handedness = None

            # Prioritize a "Right" hand if available for cursor control, or take the first detected
            if latest_hand_landmarks_data:
                for handedness, landmarks in latest_hand_landmarks_data:
                    if handedness.lower() == "right": # Or "Left" if you prefer
                        active_hand_landmarks = landmarks
                        active_handedness = handedness
                        break
                if not active_hand_landmarks: # If no "Right" hand, take the first one
                    active_handedness, active_hand_landmarks = latest_hand_landmarks_data[0]


            # --- 1. Cursor Control (using index finger tip of the active hand) ---
            if active_hand_landmarks:
                index_tip = active_hand_landmarks[INDEX_FINGER_TIP]
                thumb_tip = active_hand_landmarks[THUMB_TIP]

                # Map normalized coordinates to screen coordinates
                # Invert X because the frame is flipped
                target_cursor_x = int((1.0 - index_tip.x) * SCREEN_WIDTH)
                target_cursor_y = int(index_tip.y * SCREEN_HEIGHT)

                # Smoothing
                current_cursor_x = int(prev_cursor_x * (1 - CURSOR_SMOOTHING) + target_cursor_x * CURSOR_SMOOTHING)
                current_cursor_y = int(prev_cursor_y * (1 - CURSOR_SMOOTHING) + target_cursor_y * CURSOR_SMOOTHING)
                
                pyautogui.moveTo(current_cursor_x, current_cursor_y, duration=0) # Move mouse
                
                prev_cursor_x, prev_cursor_y = current_cursor_x, current_cursor_y


                # --- 2. Pinch to Click (distance between index and thumb) ---
                pinch_dist = calculate_distance(index_tip, thumb_tip)
                # print(f"Pinch dist: {pinch_dist:.3f}") # For debugging threshold

                if pinch_dist < PINCH_THRESHOLD:
                    if not is_pinching_for_click:
                        print(f"{active_handedness} Hand: Pinch Detected - Click!")
                        pyautogui.click()
                        is_pinching_for_click = True
                        # Draw a circle on display frame to indicate click
                        cv2.circle(display_frame_bgr, (int(index_tip.x * display_frame_bgr.shape[1]), int(index_tip.y * display_frame_bgr.shape[0])), 15, (0,0,255), -1)

                else:
                    if is_pinching_for_click:
                         is_pinching_for_click = False


            # --- 3. Gesture-based Zoom ---
            current_time = time.time()
            if latest_gestures_data and (current_time - last_zoom_time > ZOOM_COOLDOWN):
                for handedness, gesture_name, score in latest_gestures_data:
                    # Only use gestures from the active hand for zoom, or any if no specific active hand for cursor
                    if active_handedness and handedness != active_handedness:
                        continue

                    if gesture_name == "Victory" or gesture_name == "Thumb_Up": # Or "Pointing_Up"
                        print(f"{handedness} Hand: Gesture '{gesture_name}' - Zoom In")
                        pyautogui.keyDown('ctrl') # Or 'command' on macOS
                        pyautogui.scroll(1)    # Positive for zoom in / scroll up
                        pyautogui.keyUp('ctrl')
                        last_zoom_time = current_time
                        cv2.putText(display_frame_bgr, "ZOOM IN", (50, display_frame_bgr.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        break # Process one zoom action per frame
                    elif gesture_name == "Open_Palm" or gesture_name == "Thumb_Down": # Or "Closed_Fist"
                        print(f"{handedness} Hand: Gesture '{gesture_name}' - Zoom Out")
                        pyautogui.keyDown('ctrl') # Or 'command' on macOS
                        pyautogui.scroll(-1)   # Negative for zoom out / scroll down
                        pyautogui.keyUp('ctrl')
                        last_zoom_time = current_time
                        cv2.putText(display_frame_bgr, "ZOOM OUT", (50, display_frame_bgr.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        break # Process one zoom action per frame
            
            # Display recognized gestures on screen
            y_offset_gesture = 30
            for handedness_str, gesture_name, score in latest_gestures_data:
                text = f"{handedness_str}: {gesture_name} ({score:.2f})"
                cv2.putText(display_frame_bgr, text, (10, y_offset_gesture), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA) # Yellow text
                y_offset_gesture += 30

            cv2.imshow(window_name, display_frame_bgr)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Program finished.")

if __name__ == '__main__':
    main()
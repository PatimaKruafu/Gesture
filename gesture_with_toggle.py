import mediapipe as mp
import cv2
import numpy as np
import time
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.components.processors import classifier_options as mp_classifier_options

import pyautogui
# --- CRITICAL FOR PERFORMANCE ---
pyautogui.PAUSE = 0.0 # Set pause after each PyAutoGUI call to 0 seconds
pyautogui.FAILSAFE = False 

# --- Global variables ---
latest_annotated_frame_rgb = None
latest_hand_landmarks_data = []
latest_gestures_data = []

# --- HCI Control States & NEW MODE VARIABLE ---
is_mouse_control_active = True # Start with mouse control active by default
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
PINCH_THRESHOLD = 0.06
CURSOR_SMOOTHING = 0.3
prev_cursor_x, prev_cursor_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
# For reducing pyautogui.moveTo calls
prev_pyautogui_x, prev_pyautogui_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
MIN_CURSOR_MOVE_THRESHOLD = 3 # Pixels
is_pinching_for_click = False

# --- Landmark Constants ---
INDEX_FINGER_TIP = 8
THUMB_TIP = 4
# MIDDLE_FINGER_TIP = 12 # If needed

# --- Model Paths ---
hand_landmarker_path = 'tasks/hand_landmarker.task'
gesture_recognizer_path = 'tasks/gesture_recognizer.task'

# --- MediaPipe Task Setup (BaseOptions, HandLandmarker, etc.) ---
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


# --- Result Callbacks (hand_landmarker_callback, gesture_recognizer_callback) ---
# (These remain largely the same as in your previous full version)
def hand_landmarker_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_annotated_frame_rgb, latest_hand_landmarks_data
    annotated_image_np_rgb = output_image.numpy_view().copy()
    current_hand_data = []
    if result.hand_landmarks:
        for i, hand_landmarks_list in enumerate(result.hand_landmarks):
            handedness_str = "Unknown"
            if result.handedness and len(result.handedness) > i and result.handedness[i]:
                handedness_obj = result.handedness[i][0]
                handedness_str = handedness_obj.display_name if handedness_obj.display_name else handedness_obj.category_name
            current_hand_data.append((handedness_str, hand_landmarks_list))
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks_list
            ])
            mp_drawing.draw_landmarks(
                annotated_image_np_rgb, hand_landmarks_proto, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
    latest_hand_landmarks_data = current_hand_data
    latest_annotated_frame_rgb = annotated_image_np_rgb

def gesture_recognizer_callback(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_gestures_data
    
    GESTURE_SCORE_THRESHOLD = 0.7 # Define your score threshold here

    current_gestures = []
    if result.gestures:
        for i, gesture_categories_for_hand in enumerate(result.gestures): 
            if gesture_categories_for_hand: 
                for gesture_category in gesture_categories_for_hand: # Iterate through all categories for the hand
                    if gesture_category.score >= GESTURE_SCORE_THRESHOLD:
                        handedness_str = "Unknown"
                        if result.handedness and len(result.handedness) > i and result.handedness[i]:
                            handedness_obj = result.handedness[i][0] 
                            handedness_str = handedness_obj.display_name if handedness_obj.display_name else handedness_obj.category_name
                        
                        current_gestures.append((handedness_str, gesture_category.category_name, gesture_category.score))
                        break # Take the first gesture for this hand that meets the threshold
    latest_gestures_data = current_gestures

# --- Options for Tasks ---
hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_landmarker_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=hand_landmarker_callback
    )

gesture_options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=gesture_recognizer_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=gesture_recognizer_callback,
    #canned_gestures_classifier_options=mp_classifier_options.ClassifierOptions(score_threshold=0.7)
    )

# --- Helper function to calculate distance ---
def calculate_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

# --- Main Program ---
def main():
    global latest_annotated_frame_rgb, latest_hand_landmarks_data, latest_gestures_data
    global prev_cursor_x, prev_cursor_y, is_pinching_for_click
    global is_mouse_control_active # Make it accessible
    global prev_pyautogui_x, prev_pyautogui_y


    with HandLandmarker.create_from_options(hand_options) as landmarker, \
         GestureRecognizer.create_from_options(gesture_options) as recognizer:

        cap = cv2.VideoCapture(0)
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution for better performance
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        window_name = 'MediaPipe HCI Control (Press M to toggle mouse control, Q to quit)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        last_zoom_time = 0
        ZOOM_COOLDOWN = 0.5 
        prev_frame_time = 0
        new_frame_time = 0
        frame_skip_counter = 0
        PROCESS_EVERY_N_FRAMES = 1 # Adjust if needed

        while cap.isOpened():
            ret, frame_bgr_original = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                break
            
            frame_bgr_original = cv2.flip(frame_bgr_original, 1)

            frame_skip_counter += 1
            if frame_skip_counter % PROCESS_EVERY_N_FRAMES == 0:
                frame_timestamp_ms = int(time.time() * 1000)
                frame_rgb = cv2.cvtColor(frame_bgr_original, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                landmarker.detect_async(mp_image, frame_timestamp_ms)
                recognizer.recognize_async(mp_image, frame_timestamp_ms)
            
            display_frame_bgr = frame_bgr_original.copy() 
            if latest_annotated_frame_rgb is not None:
                display_frame_bgr = cv2.cvtColor(latest_annotated_frame_rgb, cv2.COLOR_RGB2BGR)

            # --- HCI Logic ---
            active_hand_landmarks = None
            active_handedness = None
            if latest_hand_landmarks_data:
                for handedness, landmarks in latest_hand_landmarks_data:
                    if handedness.lower() == "right": # Or your preferred controlling hand
                        active_hand_landmarks = landmarks
                        active_handedness = handedness
                        break
                if not active_hand_landmarks and latest_hand_landmarks_data: # Fallback
                    active_handedness, active_hand_landmarks = latest_hand_landmarks_data[0]

            # --- Conditionally execute PyAutoGUI actions ---
            if is_mouse_control_active:
                if active_hand_landmarks:
                    index_tip = active_hand_landmarks[INDEX_FINGER_TIP]
                    thumb_tip = active_hand_landmarks[THUMB_TIP]

                    target_cursor_x = int((1.0 - index_tip.x) * SCREEN_WIDTH)
                    target_cursor_y = int(index_tip.y * SCREEN_HEIGHT)
                    current_cursor_x = int(prev_cursor_x * (1 - CURSOR_SMOOTHING) + target_cursor_x * CURSOR_SMOOTHING)
                    current_cursor_y = int(prev_cursor_y * (1 - CURSOR_SMOOTHING) + target_cursor_y * CURSOR_SMOOTHING)
                    
                    # Only call moveTo if significant change
                    if abs(current_cursor_x - prev_pyautogui_x) > MIN_CURSOR_MOVE_THRESHOLD or \
                       abs(current_cursor_y - prev_pyautogui_y) > MIN_CURSOR_MOVE_THRESHOLD:
                        pyautogui.moveTo(current_cursor_x, current_cursor_y, duration=0)
                        prev_pyautogui_x = current_cursor_x
                        prev_pyautogui_y = current_cursor_y
                    
                    prev_cursor_x, prev_cursor_y = current_cursor_x, current_cursor_y # For internal smoothing

                    pinch_dist = calculate_distance(index_tip, thumb_tip)
                    if pinch_dist < PINCH_THRESHOLD:
                        if not is_pinching_for_click:
                            print(f"{active_handedness} Hand: Pinch Detected - Click!")
                            pyautogui.click()
                            is_pinching_for_click = True
                            cv2.circle(display_frame_bgr, (int(index_tip.x * display_frame_bgr.shape[1]), int(index_tip.y * display_frame_bgr.shape[0])), 15, (0,0,255), -1)
                    else:
                        if is_pinching_for_click:
                             is_pinching_for_click = False

                current_time_for_zoom = time.time()
                if latest_gestures_data and (current_time_for_zoom - last_zoom_time > ZOOM_COOLDOWN):
                    for handedness, gesture_name, score in latest_gestures_data:
                        if active_handedness and handedness != active_handedness:
                            continue # Only active hand controls zoom if one is designated for cursor
                        
                        zoom_action_taken = False
                        if gesture_name == "Victory" or gesture_name == "Thumb_Up":
                            print(f"{handedness} Hand: Gesture '{gesture_name}' - Zoom In")
                            pyautogui.keyDown('ctrl') 
                            pyautogui.scroll(1)    
                            pyautogui.keyUp('ctrl')
                            last_zoom_time = current_time_for_zoom
                            cv2.putText(display_frame_bgr, "ZOOM IN", (50, display_frame_bgr.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                            zoom_action_taken = True
                        elif gesture_name == "Open_Palm" or gesture_name == "Thumb_Down":
                            print(f"{handedness} Hand: Gesture '{gesture_name}' - Zoom Out")
                            pyautogui.keyDown('ctrl') 
                            pyautogui.scroll(-1)   
                            pyautogui.keyUp('ctrl')
                            last_zoom_time = current_time_for_zoom
                            cv2.putText(display_frame_bgr, "ZOOM OUT", (50, display_frame_bgr.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                            zoom_action_taken = True
                        
                        if zoom_action_taken:
                            break 
            
            # --- Display gesture names (always, regardless of mouse control mode) ---
            y_offset_gesture = 60 # Moved down to make space for mode text
            for handedness_str, gesture_name, score in latest_gestures_data:
                text = f"{handedness_str}: {gesture_name} ({score:.2f})"
                cv2.putText(display_frame_bgr, text, (10, y_offset_gesture), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                y_offset_gesture += 30

            # --- Display FPS ---
            new_frame_time = time.time()
            if (new_frame_time - prev_frame_time) > 0:
                fps = 1 / (new_frame_time - prev_frame_time)
            else:
                fps = 0 
            prev_frame_time = new_frame_time
            cv2.putText(display_frame_bgr, f"FPS: {fps:.2f}", (display_frame_bgr.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # --- Display Mouse Control Mode ---
            mode_text = f"Mouse Control: {'ACTIVE' if is_mouse_control_active else 'INACTIVE'} (Press M)"
            text_color = (0, 255, 0) if is_mouse_control_active else (0, 0, 255)
            cv2.putText(display_frame_bgr, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            
            cv2.imshow(window_name, display_frame_bgr)

            # --- Key press handling ---
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'): # 'm' key to toggle mouse control
                is_mouse_control_active = not is_mouse_control_active
                print(f"Mouse control {'ACTIVATED' if is_mouse_control_active else 'DEACTIVATED'}")
                # Reset pinch state when mode changes to avoid sticky clicks
                is_pinching_for_click = False 

        cap.release()
        cv2.destroyAllWindows()
        print("Program finished.")

if __name__ == '__main__':
    main()
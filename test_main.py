import mediapipe as mp
import cv2
import numpy as np
import time
from mediapipe.framework.formats import landmark_pb2
# from mediapipe.tasks.python.components.processors import classifier_options as mp_classifier_options # Only if using for gesture_options

# --- PyAutoGUI ---
import pyautogui
pyautogui.PAUSE = 0.0
pyautogui.FAILSAFE = False

# --- Global variables for MediaPipe results ---
latest_annotated_frame_rgb = None
latest_hand_landmarks_data = []   # List of (handedness, landmarks_list)
latest_gestures_data = []         # For any pre-trained gesture model if still used

# --- Model Paths ---
hand_landmarker_path = 'tasks/hand_landmarker.task'
# gesture_recognizer_path = 'tasks/gesture_recognizer.task' # Assuming we're not using the generic gesture recognizer for these custom actions for now

# --- MediaPipe Task Setup ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
# If you decide to integrate a custom gesture model later for "Dragon Claw":
# GestureRecognizer = mp.tasks.vision.GestureRecognizer
# GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
# GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# --- Landmark Constants (adjust indices as needed from MediaPipe hand landmark model) ---
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# --- HCI Control States & Constants ---
is_mouse_control_active = True      # Master toggle (M key)
is_dragon_claw_active = False     # For cursor control
is_cursor_moving_allowed = False  # Combined state: master toggle AND dragon claw

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
CURSOR_SMOOTHING = 0.2  # Adjust for desired responsiveness
prev_cursor_x, prev_cursor_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
prev_pyautogui_x, prev_pyautogui_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
MIN_CURSOR_MOVE_THRESHOLD = 2 # Pixels

# Dragon Claw Heuristic Thresholds (These will require a LOT of tuning)
DRAGON_CLAW_FINGER_DIST_THRESHOLD = 0.08 # Max distance between tips of thumb, index, middle for "claw"
DRAGON_CLAW_THUMB_BEND_ANGLE_MIN = 130 # Min angle for thumb bend (degrees)
DRAGON_CLAW_FINGER_BEND_ANGLE_MIN = 100 # Min angle for index/middle finger bend

# Finger Flick Click States & Thresholds
# Index Finger
prev_index_tip_y = 0.5
index_flick_state = "NONE" # NONE, RISING_EDGE, FALLING_EDGE
index_flick_start_time = 0
INDEX_FLICK_Y_DELTA_THRESHOLD = 0.025  # Normalized Y movement for a flick component
INDEX_FLICK_TIME_LIMIT_MS = 250     # Max time for a complete flick (milliseconds)

# Middle Finger
prev_middle_tip_y = 0.5
middle_flick_state = "NONE" # NONE, RISING_EDGE, FALLING_EDGE
middle_flick_start_time = 0
MIDDLE_FLICK_Y_DELTA_THRESHOLD = 0.025
MIDDLE_FLICK_TIME_LIMIT_MS = 250

last_left_click_time = 0
last_right_click_time = 0
CLICK_COOLDOWN_MS = 300 # Milliseconds

# --- Hand Landmarker Callback ---
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
            current_hand_data.append((handedness_str, hand_landmarks_list, timestamp_ms)) # Add timestamp_ms

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


# --- Hand Landmarker Options ---
hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_landmarker_path),
    running_mode=VisionRunningMode.LIVE_STREAM, num_hands=1, # FOCUS ON ONE HAND FOR SIMPLICITY
    min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5, result_callback=hand_landmarker_callback)


# --- Helper Functions for Geometry ---
def get_landmark(landmarks, index):
    """Safely gets a landmark or returns a dummy if out of bounds (should not happen with valid data)."""
    if 0 <= index < len(landmarks):
        return landmarks[index]
    return mp.tasks.vision.NormalizedLandmark(x=0,y=0,z=0) # Should ideally not happen

def calculate_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def calculate_angle(p1, p2, p3): # Calculates angle at p2
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    angle = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
    return angle

# --- Heuristic for Dragon Claw ---
def is_dragon_claw_gesture(landmarks):
    if not landmarks or len(landmarks) < PINKY_TIP + 1:
        return False

    thumb_tip = get_landmark(landmarks,THUMB_TIP)
    index_tip = get_landmark(landmarks,INDEX_TIP)
    middle_tip = get_landmark(landmarks,MIDDLE_TIP)
    ring_tip = get_landmark(landmarks,RING_TIP)   # For checking if they are away
    pinky_tip = get_landmark(landmarks,PINKY_TIP) # For checking if they are away

    # 1. Check distances between thumb, index, middle tips
    dist_thumb_index = calculate_distance(thumb_tip, index_tip)
    dist_thumb_middle = calculate_distance(thumb_tip, middle_tip)
    dist_index_middle = calculate_distance(index_tip, middle_tip)

    if dist_thumb_index > DRAGON_CLAW_FINGER_DIST_THRESHOLD or \
       dist_thumb_middle > DRAGON_CLAW_FINGER_DIST_THRESHOLD or \
       dist_index_middle > DRAGON_CLAW_FINGER_DIST_THRESHOLD :
        return False

    # 2. Check bend angles (rough estimate - for more accuracy, use PIP-MCP-Wrist or similar)
    # Angle at MCP joint (e.g., for thumb: THUMB_CMC, THUMB_MCP, THUMB_IP)
    # Angle for fingers usually MCP -> PIP -> DIP. Let's use MCP as pivot for a simple bend check.
    wrist = get_landmark(landmarks,WRIST)
    thumb_mcp = get_landmark(landmarks,THUMB_MCP)
    index_mcp = get_landmark(landmarks,INDEX_MCP)
    middle_mcp = get_landmark(landmarks,MIDDLE_MCP)
    
    # Angle: WRIST-MCP-PIP to see how much the finger is bent "inward"
    index_pip = get_landmark(landmarks, INDEX_PIP)
    middle_pip = get_landmark(landmarks, MIDDLE_PIP)
    thumb_ip = get_landmark(landmarks, THUMB_IP)


    try:
        # Thumb: Angle formed by WRIST, MCP, IP (or CMC, MCP, IP)
        # A smaller angle means more bent. Let's use 180 - angle to get "bend amount"
        # For thumb, let's check angle MCP-IP-TIP (more curl) or a simpler one like CMC-MCP-IP
        thumb_angle = calculate_angle(get_landmark(landmarks,THUMB_CMC), thumb_mcp, thumb_ip) # Smaller angle = more curl
        index_angle = calculate_angle(wrist, index_mcp, index_pip) # Smaller angle = more bent towards palm
        middle_angle = calculate_angle(wrist, middle_mcp, middle_pip)

        # print(f"Angles: T:{thumb_angle:.1f} I:{index_angle:.1f} M:{middle_angle:.1f}")

        # For these angles, a smaller value means MORE bent/curled.
        # So we want them to be LESS than a threshold that represents "straight"
        # For example, if 180 is straight, we want angles < 150-160 for a bend
        if thumb_angle > (180 - DRAGON_CLAW_THUMB_BEND_ANGLE_MIN) or \
           index_angle > (180 - DRAGON_CLAW_FINGER_BEND_ANGLE_MIN) or \
           middle_angle > (180 - DRAGON_CLAW_FINGER_BEND_ANGLE_MIN):
            # If angles are too large (fingers too straight), it's not a claw
            return False # This logic needs refinement based on what calculate_angle returns

        # Refined Angle Logic: Let's aim for a positive "bend" value from straight (0 deg) to fully bent (e.g. 90-130 deg)
        # Angle between vector MCP-PIP and PIP-TIP. Straighter = closer to 180. More bent = smaller.
        angle_thumb_curl = calculate_angle(get_landmark(landmarks,THUMB_MCP), get_landmark(landmarks,THUMB_IP), thumb_tip)
        angle_index_curl = calculate_angle(index_mcp, index_pip, index_tip)
        angle_middle_curl = calculate_angle(middle_mcp, middle_pip, middle_tip)
        
        # We want these curl angles to be LESS than, say, 160 (not fully straight)
        # print(f"Curl Angles: T:{angle_thumb_curl:.1f} I:{angle_index_curl:.1f} M:{angle_middle_curl:.1f}")
        if angle_thumb_curl > 150 or angle_index_curl > 160 or angle_middle_curl > 160:
             return False


    except ValueError: # Arccos domain error if points are collinear
        # print("Angle calculation error, likely collinear points")
        return False


    # 3. Optional: Check if Ring and Pinky are further away or more curled
    # This makes the heuristic more robust but also more complex.
    # For now, let's keep it simpler.

    return True # If all above conditions pass


# --- Main Program ---
def main():
    global latest_annotated_frame_rgb, latest_hand_landmarks_data
    global prev_cursor_x, prev_cursor_y, prev_pyautogui_x, prev_pyautogui_y
    global is_mouse_control_active, is_dragon_claw_active, is_cursor_moving_allowed

    global prev_index_tip_y, index_flick_state, index_flick_start_time
    global prev_middle_tip_y, middle_flick_state, middle_flick_start_time
    global last_left_click_time, last_right_click_time

    # Initialize with only HandLandmarker
    with HandLandmarker.create_from_options(hand_options) as landmarker:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        window_name = 'Advanced HCI Control (M: Toggle, Q: Quit)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        prev_frame_time = 0
        new_frame_time = 0

        while cap.isOpened():
            ret, frame_bgr_original = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                break
            
            frame_bgr_original = cv2.flip(frame_bgr_original, 1)
            current_frame_timestamp_ms = int(time.time() * 1000) # Timestamp for this iteration

            frame_rgb = cv2.cvtColor(frame_bgr_original, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            landmarker.detect_async(mp_image, current_frame_timestamp_ms) # Use consistent timestamp
            
            display_frame_bgr = frame_bgr_original.copy() 
            if latest_annotated_frame_rgb is not None:
                display_frame_bgr = cv2.cvtColor(latest_annotated_frame_rgb, cv2.COLOR_RGB2BGR)

            # --- Process Latest Hand Data ---
            active_hand_landmarks = None
            active_handedness = None
            landmarks_timestamp = 0 # Timestamp of the landmark data

            if latest_hand_landmarks_data: # Ensure it's not empty
                # For now, assuming one hand, taking the first available.
                # If using num_hands=1, latest_hand_landmarks_data will have at most one entry.
                active_handedness, active_hand_landmarks, landmarks_timestamp = latest_hand_landmarks_data[0]

            # --- Dragon Claw Detection ---
            if active_hand_landmarks:
                is_dragon_claw_active = is_dragon_claw_gesture(active_hand_landmarks)
            else:
                is_dragon_claw_active = False
            
            is_cursor_moving_allowed = is_mouse_control_active and is_dragon_claw_active

            # --- HCI Logic ---
            if active_hand_landmarks:
                # --- 1. Cursor Control (Dragon Claw based) ---
                if is_cursor_moving_allowed:
                    # Use a consistent point for cursor, e.g., midpoint of thumb, index, middle finger tips
                    # or simply index finger tip if it's stable enough with the claw.
                    # For simplicity, let's use a point near the center of the "claw"
                    # Average of Thumb, Index, Middle MCP joints could be a stable anchor point
                    
                    # Or just index tip for now, as before
                    index_tip = get_landmark(active_hand_landmarks, INDEX_TIP) 

                    target_cursor_x = int((1.0 - index_tip.x) * SCREEN_WIDTH)
                    target_cursor_y = int(index_tip.y * SCREEN_HEIGHT)
                    current_cursor_x = int(prev_cursor_x * (1 - CURSOR_SMOOTHING) + target_cursor_x * CURSOR_SMOOTHING)
                    current_cursor_y = int(prev_cursor_y * (1 - CURSOR_SMOOTHING) + target_cursor_y * CURSOR_SMOOTHING)
                    
                    if abs(current_cursor_x - prev_pyautogui_x) > MIN_CURSOR_MOVE_THRESHOLD or \
                       abs(current_cursor_y - prev_pyautogui_y) > MIN_CURSOR_MOVE_THRESHOLD:
                        pyautogui.moveTo(current_cursor_x, current_cursor_y, duration=0)
                        prev_pyautogui_x = current_cursor_x
                        prev_pyautogui_y = current_cursor_y
                    prev_cursor_x, prev_cursor_y = current_cursor_x, current_cursor_y
                
                # --- 2. Finger Flick Clicks (Only if mouse control is active, independent of Dragon Claw for click activation) ---
                if is_mouse_control_active:
                    # Current Y positions (vertical movement in image is often flick-like)
                    current_index_tip_y = get_landmark(active_hand_landmarks, INDEX_TIP).y
                    current_middle_tip_y = get_landmark(active_hand_landmarks, MIDDLE_TIP).y
                    delta_time_ms = current_frame_timestamp_ms - landmarks_timestamp # Time since landmark data was generated

                    # Left Click (Index Finger Flick)
                    if current_frame_timestamp_ms - last_left_click_time > CLICK_COOLDOWN_MS:
                        if index_flick_state == "NONE":
                            if prev_index_tip_y - current_index_tip_y > INDEX_FLICK_Y_DELTA_THRESHOLD: # Finger moved UP (y decreases)
                                index_flick_state = "RISING_EDGE"
                                index_flick_start_time = current_frame_timestamp_ms
                        elif index_flick_state == "RISING_EDGE":
                            if current_frame_timestamp_ms - index_flick_start_time > INDEX_FLICK_TIME_LIMIT_MS:
                                index_flick_state = "NONE" # Timeout
                            elif current_index_tip_y - prev_index_tip_y > INDEX_FLICK_Y_DELTA_THRESHOLD: # Finger moved DOWN
                                print("LEFT CLICK (Index Flick)")
                                pyautogui.click(button='left')
                                index_flick_state = "NONE"
                                last_left_click_time = current_frame_timestamp_ms
                                cv2.circle(display_frame_bgr, (int(get_landmark(active_hand_landmarks, INDEX_TIP).x * display_frame_bgr.shape[1]),
                                                                int(current_index_tip_y * display_frame_bgr.shape[0])), 15, (0,255,0), -1)

                    # Right Click (Middle Finger Flick) - Similar logic
                    if current_frame_timestamp_ms - last_right_click_time > CLICK_COOLDOWN_MS:
                        if middle_flick_state == "NONE":
                            if prev_middle_tip_y - current_middle_tip_y > MIDDLE_FLICK_Y_DELTA_THRESHOLD: # Finger moved UP
                                middle_flick_state = "RISING_EDGE"
                                middle_flick_start_time = current_frame_timestamp_ms
                        elif middle_flick_state == "RISING_EDGE":
                            if current_frame_timestamp_ms - middle_flick_start_time > MIDDLE_FLICK_TIME_LIMIT_MS:
                                middle_flick_state = "NONE" # Timeout
                            elif current_middle_tip_y - prev_middle_tip_y > MIDDLE_FLICK_Y_DELTA_THRESHOLD: # Finger moved DOWN
                                print("RIGHT CLICK (Middle Flick)")
                                pyautogui.click(button='right')
                                middle_flick_state = "NONE"
                                last_right_click_time = current_frame_timestamp_ms
                                cv2.circle(display_frame_bgr, (int(get_landmark(active_hand_landmarks, MIDDLE_TIP).x * display_frame_bgr.shape[1]),
                                                                int(current_middle_tip_y * display_frame_bgr.shape[0])), 15, (0,128,255), -1) # Orange circle

                    prev_index_tip_y = current_index_tip_y
                    prev_middle_tip_y = current_middle_tip_y
            else: # No hand landmarks
                is_dragon_claw_active = False
                is_cursor_moving_allowed = False
                index_flick_state = "NONE" # Reset flick states if hand is lost
                middle_flick_state = "NONE"


            # --- UI Display ---
            new_frame_time = time.time()
            if (new_frame_time - prev_frame_time) > 0: fps = 1 / (new_frame_time - prev_frame_time)
            else: fps = 0 
            prev_frame_time = new_frame_time
            cv2.putText(display_frame_bgr, f"FPS: {fps:.2f}", (display_frame_bgr.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            mode_text = f"Mouse Ctrl: {'ON' if is_mouse_control_active else 'OFF'} (M)"
            cv2.putText(display_frame_bgr, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if is_mouse_control_active else (0,0,255), 2)
            
            claw_text = f"Claw Active: {'YES' if is_dragon_claw_active else 'NO'}"
            cv2.putText(display_frame_bgr, claw_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0) if is_dragon_claw_active else (128,128,128), 2)
            
            cursor_mode_text = f"Cursor Moving: {'ALLOWED' if is_cursor_moving_allowed else 'BLOCKED'}"
            cv2.putText(display_frame_bgr, cursor_mode_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if is_cursor_moving_allowed else (0,0,255), 2)


            cv2.imshow(window_name, display_frame_bgr)
            key = cv2.waitKey(1) & 0xFF # Reduced waitKey delay for higher responsiveness
            if key == ord('q'): break
            elif key == ord('m'):
                is_mouse_control_active = not is_mouse_control_active
                print(f"Mouse control {'ACTIVATED' if is_mouse_control_active else 'DEACTIVATED'}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("Program finished.")

if __name__ == '__main__':
    main()
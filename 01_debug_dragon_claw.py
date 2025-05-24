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

FINGER_TAP_DOWN_Y_DELTA_THRESHOLD = 0.03 # How much finger tip Y must increase


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
# Make sure these global constants are defined or passed if not global
# DRAGON_CLAW_FINGER_DIST_THRESHOLD = ... # Needs to be set

def is_dragon_claw_gesture(landmarks):
    # Global thresholds or pass them as arguments if you prefer
    # These are EXAMPLE values, you MUST set them based on YOUR data for the INTENDED claw
    DRAGON_CLAW_FINGER_DIST_THRESHOLD = 0.25 # SIGNIFICANTLY WIDENED
    # If I-M must stay close, we might need separate thresholds:
    # TI_TM_DIST_THRESHOLD = 0.25
    # IM_DIST_THRESHOLD = 0.08
    # tips_are_tripod_close = (dist_thumb_index < TI_TM_DIST_THRESHOLD and \
    #                          dist_thumb_middle < TI_TM_DIST_THRESHOLD and \
    #                          dist_index_middle < IM_DIST_THRESHOLD)
    # For now, let's try one looser threshold:


    # --- Condition 2: Specific Curl Angles for Thumb, Index, Middle ---
    # Data shows huge variation. We need much wider windows.
    # Remember: angle < MAX (not too straight), angle > MIN (not too fisted)

    # Thumb (T): observed 137 to 178
    THUMB_CURL_MIN_ANGLE_CLAW = 120  # Was 145. Allow more curl.
    THUMB_CURL_MAX_ANGLE_CLAW = 180  # Was 155. Allow fully straight (max value anyway).

    # Index (I): observed 112 to 176
    INDEX_CURL_MIN_ANGLE_CLAW = 100  # Was 164. Allow much more curl.
    INDEX_CURL_MAX_ANGLE_CLAW = 180  # Was 172. Allow fully straight.

    # Middle (M): observed 63 to 171
    MIDDLE_CURL_MIN_ANGLE_CLAW = 60   # Was 140. Allow much more curl.
    MIDDLE_CURL_MAX_ANGLE_CLAW = 180  # Was 155. Allow fully straight.

    # ... (rest of specific_curls_met logic) ...


    # --- Condition 3: Ring and Pinky are MORE curled (tucked away) ---
    # Your R/P curls are mostly very small, but Pinky sometimes goes up.
    # If threshold was 70, sometimes pinky > 70 (e.g., 76, 80s, 90s, 100s) causes failure.
    RING_PINKY_TUCKED_MAX_ANGLE = 110 # WIDENED. Allows R/P to be less tightly tucked.
                                      # This might make it less distinct from other gestures.


    if not landmarks or len(landmarks) < PINKY_TIP + 1:
        # print("DBG DragonClaw: No landmarks")
        return False

    thumb_tip = get_landmark(landmarks, THUMB_TIP)
    index_tip = get_landmark(landmarks, INDEX_TIP)
    middle_tip = get_landmark(landmarks, MIDDLE_TIP)
    ring_pip = get_landmark(landmarks, RING_PIP)
    ring_tip = get_landmark(landmarks, RING_TIP)
    pinky_pip = get_landmark(landmarks, PINKY_PIP)
    pinky_tip = get_landmark(landmarks, PINKY_TIP)

    # Condition 1: Tip distances
    dist_thumb_index = calculate_distance(thumb_tip, index_tip)
    dist_thumb_middle = calculate_distance(thumb_tip, middle_tip)
    dist_index_middle = calculate_distance(index_tip, middle_tip)
    print(f"DBG ACTUAL DISTS: T-I: {dist_thumb_index:.4f}, T-M: {dist_thumb_middle:.4f}, I-M: {dist_index_middle:.4f} (Threshold for ALL: {DRAGON_CLAW_FINGER_DIST_THRESHOLD})")
    tips_are_tripod_close = (dist_thumb_index < DRAGON_CLAW_FINGER_DIST_THRESHOLD and
                             dist_thumb_middle < DRAGON_CLAW_FINGER_DIST_THRESHOLD and
                             dist_index_middle < DRAGON_CLAW_FINGER_DIST_THRESHOLD)
    print(f"DBG tips_are_tripod_close: {tips_are_tripod_close}")

    # Condition 2: Specific Curl Angles for Thumb, Index, Middle
    specific_curls_met = False 
    try:
        thumb_mcp = get_landmark(landmarks, THUMB_MCP)
        thumb_ip = get_landmark(landmarks, THUMB_IP)
        index_mcp = get_landmark(landmarks, INDEX_MCP)
        index_pip = get_landmark(landmarks, INDEX_PIP)
        middle_mcp = get_landmark(landmarks, MIDDLE_MCP)
        middle_pip = get_landmark(landmarks, MIDDLE_PIP)

        angle_thumb_curl = calculate_angle(thumb_mcp, thumb_ip, thumb_tip)
        angle_index_curl = calculate_angle(index_mcp, index_pip, index_tip)
        angle_middle_curl = calculate_angle(middle_mcp, middle_pip, middle_tip)
        print(f"DBG TARGET CLAW? Curl Angles (MCP-PIP/IP-TIP): T:{angle_thumb_curl:.1f}, I:{angle_index_curl:.1f}, M:{angle_middle_curl:.1f}")

        thumb_in_range = (angle_thumb_curl > THUMB_CURL_MIN_ANGLE_CLAW and angle_thumb_curl < THUMB_CURL_MAX_ANGLE_CLAW)
        index_in_range = (angle_index_curl > INDEX_CURL_MIN_ANGLE_CLAW and angle_index_curl < INDEX_CURL_MAX_ANGLE_CLAW)
        middle_in_range = (angle_middle_curl > MIDDLE_CURL_MIN_ANGLE_CLAW and angle_middle_curl < MIDDLE_CURL_MAX_ANGLE_CLAW)
        specific_curls_met = thumb_in_range and index_in_range and middle_in_range
        print(f"DBG specific_curls_met: {specific_curls_met} (T:{thumb_in_range}, I:{index_in_range}, M:{middle_in_range})")
    except Exception as e:
        print(f"DBG Angle calc error: {e}")
        return False

    # Condition 3: Ring and Pinky are MORE curled
    ring_pinky_tucked = False
    try:
        ring_mcp = get_landmark(landmarks, RING_MCP)
        pinky_mcp = get_landmark(landmarks, PINKY_MCP)
        angle_ring_curl = calculate_angle(ring_mcp, ring_pip, ring_tip)
        angle_pinky_curl = calculate_angle(pinky_mcp, pinky_pip, pinky_tip)
        print(f"DBG Ring/Pinky Curls: R:{angle_ring_curl:.1f}, P:{angle_pinky_curl:.1f} (Threshold < {RING_PINKY_TUCKED_MAX_ANGLE})")
        ring_pinky_tucked = (angle_ring_curl < RING_PINKY_TUCKED_MAX_ANGLE and \
                             angle_pinky_curl < RING_PINKY_TUCKED_MAX_ANGLE)
        print(f"DBG ring_pinky_tucked: {ring_pinky_tucked}")
    except Exception as e:
        print(f"DBG Ring/Pinky Angle error: {e}")
        pass 

    is_claw = tips_are_tripod_close and specific_curls_met and ring_pinky_tucked 
    print(f"DBG FINAL CLAW: {is_claw} (TipsClose:{tips_are_tripod_close}, SpecificCurls:{specific_curls_met}, RingPinkyTucked:{ring_pinky_tucked})")
    return is_claw


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
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
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
                # --- 1. Cursor Control (Dragon Claw based, NOW USING WRIST) ---
                if is_cursor_moving_allowed:
                    # === CHANGE HERE: Use WRIST landmark instead of INDEX_TIP ===
                    palm_anchor_point = get_landmark(active_hand_landmarks, WRIST) 
                    # You could also try:
                    # palm_anchor_point = get_landmark(active_hand_landmarks, MIDDLE_MCP)
                    # Or calculate a center:
                    # mcp_index = get_landmark(active_hand_landmarks, INDEX_MCP)
                    # mcp_middle = get_landmark(active_hand_landmarks, MIDDLE_MCP)
                    # mcp_ring = get_landmark(active_hand_landmarks, RING_MCP)
                    # palm_anchor_point_x = (mcp_index.x + mcp_middle.x + mcp_ring.x) / 3
                    # palm_anchor_point_y = (mcp_index.y + mcp_middle.y + mcp_ring.y) / 3
                    # palm_anchor_point_z = (mcp_index.z + mcp_middle.z + mcp_ring.z) / 3 # if needed
                    # palm_anchor_point = mp.tasks.vision.NormalizedLandmark(x=palm_anchor_point_x, y=palm_anchor_point_y, z=palm_anchor_point_z)


                    target_cursor_x = int(palm_anchor_point.x * SCREEN_WIDTH)
                    target_cursor_y = int(palm_anchor_point.y * SCREEN_HEIGHT) 
                    # === END CHANGE FOR CURSOR ANCHOR ===
                    
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
                    current_index_tip_y = get_landmark(active_hand_landmarks, INDEX_TIP).y
                    current_middle_tip_y = get_landmark(active_hand_landmarks, MIDDLE_TIP).y
                    # current_frame_timestamp_ms is already available from your loop

                    # --- Left Click (Index Finger Tap Down) ---
                    if current_frame_timestamp_ms - last_left_click_time > CLICK_COOLDOWN_MS:
                        # Check for DOWNWARD movement (current Y > previous Y by a threshold)
                        if current_index_tip_y - prev_index_tip_y > FINGER_TAP_DOWN_Y_DELTA_THRESHOLD:
                            print("LEFT CLICK (Index Tap Down)")
                            pyautogui.click(button='left')
                            last_left_click_time = current_frame_timestamp_ms
                            cv2.circle(display_frame_bgr, (int(get_landmark(active_hand_landmarks, INDEX_TIP).x * display_frame_bgr.shape[1]),
                                                            int(current_index_tip_y * display_frame_bgr.shape[0])), 15, (0,255,0), -1)
                            # After a click, we might want to update prev_index_tip_y to the current
                            # position to prevent immediate re-triggering if the finger stays down
                            # and bounces slightly above and below the threshold again quickly.
                            # However, the CLICK_COOLDOWN_MS largely handles this.
                            # Let's update it to require a move "up" again before another tap.
                            # For this to work well, prev_index_tip_y should ideally represent a more "neutral" or "up" position.
                            # So, we might only update prev_index_tip_y if the finger is not moving down significantly.

                    # --- Right Click (Middle Finger Tap Down) ---
                    if current_frame_timestamp_ms - last_right_click_time > CLICK_COOLDOWN_MS:
                        # Check for DOWNWARD movement
                        if current_middle_tip_y - prev_middle_tip_y > FINGER_TAP_DOWN_Y_DELTA_THRESHOLD:
                            print("RIGHT CLICK (Middle Tap Down)")
                            pyautogui.click(button='right')
                            last_right_click_time = current_frame_timestamp_ms
                            cv2.circle(display_frame_bgr, (int(get_landmark(active_hand_landmarks, MIDDLE_TIP).x * display_frame_bgr.shape[1]),
                                                            int(current_middle_tip_y * display_frame_bgr.shape[0])), 15, (0,128,255), -1)
                    
                    # Update previous Y positions for the next frame's comparison.
                    # This is a crucial part: when do we consider the finger "reset" for the next tap?
                    # Option A: Always update (simplest, relies on cooldown and Y_DELTA threshold)
                    prev_index_tip_y = current_index_tip_y
                    prev_middle_tip_y = current_middle_tip_y

                    # Option B: Only update if not moving down sharply, or if moving up.
                    # This tries to make 'prev_y' represent a more "ready" or "up" state.
                    # If finger is still moving down, prev_y won't update, preventing chained clicks from one long down motion.
                    # if not (current_index_tip_y - prev_index_tip_y > FINGER_TAP_DOWN_Y_DELTA_THRESHOLD * 0.5): # Not significantly moving down
                    #     prev_index_tip_y = current_index_tip_y
                    # if not (current_middle_tip_y - prev_middle_tip_y > FINGER_TAP_DOWN_Y_DELTA_THRESHOLD * 0.5): # Not significantly moving down
                    #     prev_middle_tip_y = current_middle_tip_y

            else: # No hand landmarks
                # Reset previous positions if hand is lost, to avoid stale data on re-detection
                prev_index_tip_y = 0.5 
                prev_middle_tip_y = 0.5
                # is_dragon_claw_active = False # Assuming this is handled elsewhere too
                # is_cursor_moving_allowed = False

            #OLD
            """ else: # No hand landmarks
                is_dragon_claw_active = False
                is_cursor_moving_allowed = False
                index_flick_state = "NONE" # Reset flick states if hand is lost
                middle_flick_state = "NONE" """


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
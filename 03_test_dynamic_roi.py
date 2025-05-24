import mediapipe as mp
import cv2
import numpy as np
import time
from mediapipe.framework.formats import landmark_pb2

# --- PyAutoGUI ---
import pyautogui
pyautogui.PAUSE = 0.0
pyautogui.FAILSAFE = False

# --- Global variables for MediaPipe results ---
latest_annotated_frame_rgb = None
latest_hand_landmarks_data = []
# latest_gestures_data = [] # Not used

# --- Model Paths ---
hand_landmarker_path = 'tasks/hand_landmarker.task'

# --- MediaPipe Task Setup ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# --- Landmark Constants ---
WRIST = 0; THUMB_TIP = 4; INDEX_TIP = 8; MIDDLE_TIP = 12; # Simplified
THUMB_CMC = 1; THUMB_MCP = 2; THUMB_IP = 3;
INDEX_MCP = 5; INDEX_PIP = 6;
MIDDLE_MCP = 9; MIDDLE_PIP = 10;
RING_MCP = 13; RING_PIP = 14; RING_TIP = 16;
PINKY_MCP = 17; PINKY_PIP = 18; PINKY_TIP = 20;


# --- HCI Control States & Constants ---
is_mouse_control_active = True
# is_dragon_claw_active & is_cursor_moving_allowed will be set in main

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
CURSOR_SMOOTHING = 0.2
prev_cursor_x, prev_cursor_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
prev_pyautogui_x, prev_pyautogui_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
MIN_CURSOR_MOVE_THRESHOLD = 2

# Tap Click States & Thresholds
prev_index_tip_y = 0.5
prev_middle_tip_y = 0.5
last_left_click_time = 0
last_right_click_time = 0
CLICK_COOLDOWN_MS = 300
FINGER_TAP_DOWN_Y_DELTA_THRESHOLD = 0.04 # Tuned from previous discussions

# --- DYNAMIC ROI CONSTANTS (REPLACE OLD STATIC ROI_X_MIN etc.) ---
ROI_HALF_WIDTH = 0.15  # ROI will be 0.30 (30%) of camera width
ROI_HALF_HEIGHT = 0.20 # ROI will be 0.40 (40%) of camera height
ROI_POWER_X = 1.5      # Non-linearity for X
ROI_POWER_Y = 1.5      # Non-linearity for Y
# --- END DYNAMIC ROI CONSTANTS ---


# --- Hand Landmarker Callback (Unchanged) ---
def hand_landmarker_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_annotated_frame_rgb, latest_hand_landmarks_data
    annotated_image_np_rgb = output_image.numpy_view().copy()
    current_hand_data = []
    if result.hand_landmarks:
        for i, hand_landmarks_list in enumerate(result.hand_landmarks):
            handedness_str = "Unknown"; lm_has_z = hasattr(hand_landmarks_list[0], 'z')
            if result.handedness and len(result.handedness) > i and result.handedness[i]:
                handedness_obj = result.handedness[i][0]
                handedness_str = handedness_obj.display_name if handedness_obj.display_name else handedness_obj.category_name
            current_hand_data.append((handedness_str, hand_landmarks_list, timestamp_ms))
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z if lm_has_z else 0) for lm in hand_landmarks_list
            ])
            mp_drawing.draw_landmarks(
                annotated_image_np_rgb, hand_landmarks_proto, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
    latest_hand_landmarks_data = current_hand_data
    latest_annotated_frame_rgb = annotated_image_np_rgb

# --- Hand Landmarker Options (Unchanged) ---
hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_landmarker_path),
    running_mode=VisionRunningMode.LIVE_STREAM, num_hands=1,
    min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5, result_callback=hand_landmarker_callback)

# --- Helper Functions (get_landmark, calculate_distance, calculate_angle - Unchanged, but added z safety) ---
def get_landmark(landmarks, index):
    if 0 <= index < len(landmarks): return landmarks[index]
    return mp.tasks.vision.NormalizedLandmark(x=0,y=0,z=0) # Default with z

def calculate_distance(p1, p2): # Assumes p1, p2 have .x, .y, and .z (or z defaults to 0 if not present)
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    dz = getattr(p1, 'z', 0) - getattr(p2, 'z', 0) # Handle missing z
    return np.sqrt(dx**2 + dy**2 + dz**2)

def calculate_angle(p1, p2, p3): # p1,p2,p3 are NormalizedLandmark objects
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, getattr(p1, 'z', 0) - getattr(p2, 'z', 0)])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, getattr(p3, 'z', 0) - getattr(p2, 'z', 0)])
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0: return 180.0 
    cosine_angle = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

# --- Heuristic for Dragon Claw (Unchanged from your last working version) ---
def is_dragon_claw_gesture(landmarks):
    # USING THE WIDENED, WORKING THRESHOLDS FROM YOUR SUCCESSFUL LOGS
    DRAGON_CLAW_FINGER_DIST_THRESHOLD = 0.25 
    THUMB_CURL_MIN_ANGLE_CLAW = 120; THUMB_CURL_MAX_ANGLE_CLAW = 180 
    INDEX_CURL_MIN_ANGLE_CLAW = 100; INDEX_CURL_MAX_ANGLE_CLAW = 180  
    MIDDLE_CURL_MIN_ANGLE_CLAW = 60; MIDDLE_CURL_MAX_ANGLE_CLAW = 180  
    RING_PINKY_TUCKED_MAX_ANGLE = 110 
    # (Keep the rest of your is_dragon_claw_gesture logic, print statements removed for brevity here)
    if not landmarks or len(landmarks) < PINKY_TIP + 1: return False
    thumb_tip = get_landmark(landmarks, THUMB_TIP); index_tip = get_landmark(landmarks, INDEX_TIP); middle_tip = get_landmark(landmarks, MIDDLE_TIP)
    ring_pip = get_landmark(landmarks, RING_PIP); ring_tip = get_landmark(landmarks, RING_TIP); pinky_pip = get_landmark(landmarks, PINKY_PIP); pinky_tip = get_landmark(landmarks, PINKY_TIP)
    dist_thumb_index = calculate_distance(thumb_tip, index_tip); dist_thumb_middle = calculate_distance(thumb_tip, middle_tip); dist_index_middle = calculate_distance(index_tip, middle_tip)
    tips_are_tripod_close = (dist_thumb_index < DRAGON_CLAW_FINGER_DIST_THRESHOLD and dist_thumb_middle < DRAGON_CLAW_FINGER_DIST_THRESHOLD and dist_index_middle < DRAGON_CLAW_FINGER_DIST_THRESHOLD)
    specific_curls_met = False 
    try:
        thumb_mcp = get_landmark(landmarks, THUMB_MCP); thumb_ip = get_landmark(landmarks, THUMB_IP); index_mcp = get_landmark(landmarks, INDEX_MCP); index_pip = get_landmark(landmarks, INDEX_PIP); middle_mcp = get_landmark(landmarks, MIDDLE_MCP); middle_pip = get_landmark(landmarks, MIDDLE_PIP)
        angle_thumb_curl = calculate_angle(thumb_mcp, thumb_ip, thumb_tip); angle_index_curl = calculate_angle(index_mcp, index_pip, index_tip); angle_middle_curl = calculate_angle(middle_mcp, middle_pip, middle_tip)
        thumb_in_range = (angle_thumb_curl > THUMB_CURL_MIN_ANGLE_CLAW and angle_thumb_curl < THUMB_CURL_MAX_ANGLE_CLAW)
        index_in_range = (angle_index_curl > INDEX_CURL_MIN_ANGLE_CLAW and angle_index_curl < INDEX_CURL_MAX_ANGLE_CLAW)
        middle_in_range = (angle_middle_curl > MIDDLE_CURL_MIN_ANGLE_CLAW and angle_middle_curl < MIDDLE_CURL_MAX_ANGLE_CLAW)
        specific_curls_met = thumb_in_range and index_in_range and middle_in_range
    except: return False
    ring_pinky_tucked = False
    try:
        ring_mcp = get_landmark(landmarks, RING_MCP); pinky_mcp = get_landmark(landmarks, PINKY_MCP)
        angle_ring_curl = calculate_angle(ring_mcp, ring_pip, ring_tip); angle_pinky_curl = calculate_angle(pinky_mcp, pinky_pip, pinky_tip)
        ring_pinky_tucked = (angle_ring_curl < RING_PINKY_TUCKED_MAX_ANGLE and angle_pinky_curl < RING_PINKY_TUCKED_MAX_ANGLE)
    except: pass 
    return tips_are_tripod_close and specific_curls_met and ring_pinky_tucked 


# --- Main Program ---
def main():
    global latest_annotated_frame_rgb, latest_hand_landmarks_data
    global prev_cursor_x, prev_cursor_y, prev_pyautogui_x, prev_pyautogui_y
    global is_mouse_control_active # Modifiable by key press
    # These are effectively local to main's loop, but depend on globals like is_mouse_control_active
    # is_dragon_claw_active_current_frame (local)
    # can_attempt_cursor_control (local)
    global prev_index_tip_y, prev_middle_tip_y
    global last_left_click_time, last_right_click_time

    last_wrist_pos_for_roi_check = None # Initialize to None

    # --- DYNAMIC ROI INITIALIZATION ---
    current_roi_center_x = 0.5  # Start ROI in the center of camera view
    current_roi_center_y = 0.5
    last_wrist_pos_for_roi_check = None # This will store a NormalizedLandmark object
    time_wrist_stable_at_pos_start_ms = 0 # Timestamp when wrist became stable
    ROI_RECENTER_STABILITY_DURATION_MS = 2000  # 3 seconds for ROI to recenter
    ROI_RECENTER_MOVEMENT_THRESHOLD = 0.03     # Normalized distance for stability

    # --- Other Main Loop State Variables ---
    prev_is_dragon_claw_active = False 
    last_valid_target_x = SCREEN_WIDTH // 2
    last_valid_target_y = SCREEN_HEIGHT // 2
    
    with HandLandmarker.create_from_options(hand_options) as landmarker:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): print("Error: Could not open webcam."); return
        window_name = 'Dynamic ROI HCI (M: Toggle, Q: Quit)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        prev_frame_time = 0

        while cap.isOpened():
            ret, frame_bgr_original = cap.read()
            if not ret: print("Ignoring empty camera frame."); break
            
            frame_bgr_original = cv2.flip(frame_bgr_original, 1)
            current_frame_timestamp_ms = int(time.time() * 1000)

            frame_rgb = cv2.cvtColor(frame_bgr_original, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            landmarker.detect_async(mp_image, current_frame_timestamp_ms)
            
            display_frame_bgr = frame_bgr_original.copy()
            if latest_annotated_frame_rgb is not None:
                display_frame_bgr = cv2.cvtColor(latest_annotated_frame_rgb, cv2.COLOR_RGB2BGR)

            active_hand_landmarks = None
            is_dragon_claw_active_current_frame = False # Default for this frame

            if latest_hand_landmarks_data: # If mediapipe callback provided data
                _, active_hand_landmarks, _ = latest_hand_landmarks_data[0] # landmarks_list

            # --- DYNAMIC ROI AND GESTURE LOGIC ---
            if active_hand_landmarks:
                palm_anchor_point = get_landmark(active_hand_landmarks, WRIST)
                is_dragon_claw_active_current_frame = is_dragon_claw_gesture(active_hand_landmarks)

                # Dynamic ROI Re-centering
                if last_wrist_pos_for_roi_check is not None:
                    # Check distance using only X and Y for stability in plane
                    #temp_last_wrist = mp.tasks.vision.NormalizedLandmark(x=last_wrist_pos_for_roi_check.x, y=last_wrist_pos_for_roi_check.y, z=0)
                    #temp_palm_anchor = mp.tasks.vision.NormalizedLandmark(x=palm_anchor_point.x, y=palm_anchor_point.y, z=0)
                    #dist_moved = calculate_distance(temp_palm_anchor, temp_last_wrist)
                    dist_moved = calculate_distance(palm_anchor_point, last_wrist_pos_for_roi_check)

                    if dist_moved < ROI_RECENTER_MOVEMENT_THRESHOLD:
                        if time_wrist_stable_at_pos_start_ms == 0: # Just became stable
                            time_wrist_stable_at_pos_start_ms = current_frame_timestamp_ms
                        elif current_frame_timestamp_ms - time_wrist_stable_at_pos_start_ms > ROI_RECENTER_STABILITY_DURATION_MS:
                            # Re-center ROI
                            current_roi_center_x = palm_anchor_point.x
                            current_roi_center_y = palm_anchor_point.y
                            time_wrist_stable_at_pos_start_ms = 0 
                            #last_wrist_pos_for_roi_check = mp.tasks.vision.NormalizedLandmark(x=palm_anchor_point.x, y=palm_anchor_point.y, z=getattr(palm_anchor_point, 'z', 0))
                            last_wrist_pos_for_roi_check = palm_anchor_point
                            # Mitigate cursor jump when ROI re-centers
                            actual_mouse_x, actual_mouse_y = pyautogui.position()
                            prev_cursor_x, prev_pyautogui_x = actual_mouse_x, actual_mouse_x
                            prev_cursor_y, prev_pyautogui_y = actual_mouse_y, actual_mouse_y
                            last_valid_target_x, last_valid_target_y = actual_mouse_x, actual_mouse_y
                            # print(f"DBG ROI Re-centered to: ({current_roi_center_x:.2f}, {current_roi_center_y:.2f})")
                            
                    else: # Wrist moved too much
                        time_wrist_stable_at_pos_start_ms = 0
                        #last_wrist_pos_for_roi_check = mp.tasks.vision.NormalizedLandmark(x=palm_anchor_point.x, y=palm_anchor_point.y, z=getattr(palm_anchor_point, 'z', 0))
                        last_wrist_pos_for_roi_check = palm_anchor_point
                else: # First time seeing the hand
                    #last_wrist_pos_for_roi_check = mp.tasks.vision.NormalizedLandmark(x=palm_anchor_point.x, y=palm_anchor_point.y, z=getattr(palm_anchor_point, 'z', 0))
                    last_wrist_pos_for_roi_check = palm_anchor_point
                    time_wrist_stable_at_pos_start_ms = 0 
            else: # No hand landmarks
                last_wrist_pos_for_roi_check = None
                time_wrist_stable_at_pos_start_ms = 0
                is_dragon_claw_active_current_frame = False
                # Reset tap click prev_y states
                prev_index_tip_y = 0.5 
                prev_middle_tip_y = 0.5

            # Calculate current dynamic ROI boundaries AFTER potential re-centering
            clamped_roi_center_x = np.clip(current_roi_center_x, ROI_HALF_WIDTH, 1.0 - ROI_HALF_WIDTH)
            clamped_roi_center_y = np.clip(current_roi_center_y, ROI_HALF_HEIGHT, 1.0 - ROI_HALF_HEIGHT)
            dynamic_roi_x_min = clamped_roi_center_x - ROI_HALF_WIDTH
            dynamic_roi_x_max = clamped_roi_center_x + ROI_HALF_WIDTH
            dynamic_roi_y_min = clamped_roi_center_y - ROI_HALF_HEIGHT
            dynamic_roi_y_max = clamped_roi_center_y + ROI_HALF_HEIGHT

            # --- Determine if cursor control can proceed this frame ---
            can_attempt_cursor_control = is_mouse_control_active and is_dragon_claw_active_current_frame

            if not prev_is_dragon_claw_active and is_dragon_claw_active_current_frame: # Clutch
                actual_mouse_x, actual_mouse_y = pyautogui.position()
                prev_cursor_x, prev_pyautogui_x = actual_mouse_x, actual_mouse_x
                prev_cursor_y, prev_pyautogui_y = actual_mouse_y, actual_mouse_y
                last_valid_target_x, last_valid_target_y = actual_mouse_x, actual_mouse_y
                # print("DBG CLUTCH: Dragon Claw just (re)engaged.")

            # --- CURSOR MOVEMENT LOGIC ---
            target_x_for_smoothing = last_valid_target_x # Default to last valid
            target_y_for_smoothing = last_valid_target_y
            update_cursor_this_frame = False 

            if active_hand_landmarks and can_attempt_cursor_control:
                # palm_anchor_point is already defined (WRIST)
                hand_in_roi_x = (palm_anchor_point.x >= dynamic_roi_x_min and palm_anchor_point.x <= dynamic_roi_x_max)
                hand_in_roi_y = (palm_anchor_point.y >= dynamic_roi_y_min and palm_anchor_point.y <= dynamic_roi_y_max)
                is_hand_in_roi = hand_in_roi_x and hand_in_roi_y # Local variable for clarity
                
                # print(f"DBG PreROI: CtrlON={is_mouse_control_active}, ClawON={is_dragon_claw_active_current_frame}, AttemptCtrl={can_attempt_cursor_control}, InROI={is_hand_in_roi}, WristXY=({palm_anchor_point.x:.3f},{palm_anchor_point.y:.3f}) DynROI_X({dynamic_roi_x_min:.2f}-{dynamic_roi_x_max:.2f})")

                if is_hand_in_roi: # If hand is within the DYNAMIC ROI
                    update_cursor_this_frame = True
                    roi_width = dynamic_roi_x_max - dynamic_roi_x_min
                    roi_height = dynamic_roi_y_max - dynamic_roi_y_min
                    if roi_width <= 0: roi_width = 1e-6 # Avoid division by zero/small
                    if roi_height <= 0: roi_height = 1e-6

                    hand_norm_in_roi_x = np.clip((palm_anchor_point.x - dynamic_roi_x_min) / roi_width, 0.0, 1.0)
                    hand_norm_in_roi_y = np.clip((palm_anchor_point.y - dynamic_roi_y_min) / roi_height, 0.0, 1.0)

                    val_x = hand_norm_in_roi_x
                    if val_x < 0.5: screen_norm_x = 0.5 * pow(val_x * 2.0, ROI_POWER_X)
                    else: screen_norm_x = 1.0 - (0.5 * pow((1.0 - val_x) * 2.0, ROI_POWER_X))
                    screen_norm_x = np.clip(screen_norm_x, 0.0, 1.0)

                    val_y = hand_norm_in_roi_y
                    if val_y < 0.5: screen_norm_y = 0.5 * pow(val_y * 2.0, ROI_POWER_Y)
                    else: screen_norm_y = 1.0 - (0.5 * pow((1.0 - val_y) * 2.0, ROI_POWER_Y))
                    screen_norm_y = np.clip(screen_norm_y, 0.0, 1.0)
                    
                    new_target_x = int(screen_norm_x * SCREEN_WIDTH)
                    new_target_y = int(screen_norm_y * SCREEN_HEIGHT)

                    target_x_for_smoothing = new_target_x
                    target_y_for_smoothing = new_target_y
                    last_valid_target_x = new_target_x
                    last_valid_target_y = new_target_y
                    # print(f"DBG ROI-CALC: Target=({new_target_x},{new_target_y})")
                # else: print(f"DBG ROI: Hand OUTSIDE DynROI")

            current_smoothed_x = int(prev_cursor_x * (1 - CURSOR_SMOOTHING) + target_x_for_smoothing * CURSOR_SMOOTHING)
            current_smoothed_y = int(prev_cursor_y * (1 - CURSOR_SMOOTHING) + target_y_for_smoothing * CURSOR_SMOOTHING)

            if can_attempt_cursor_control:
                if abs(current_smoothed_x - prev_pyautogui_x) > MIN_CURSOR_MOVE_THRESHOLD or \
                   abs(current_smoothed_y - prev_pyautogui_y) > MIN_CURSOR_MOVE_THRESHOLD:
                    pyautogui.moveTo(current_smoothed_x, current_smoothed_y, duration=0)
                    prev_pyautogui_x = current_smoothed_x
                    prev_pyautogui_y = current_smoothed_y
            
            prev_cursor_x = current_smoothed_x
            prev_cursor_y = current_smoothed_y
            prev_is_dragon_claw_active = is_dragon_claw_active_current_frame

            # --- Click Logic ---
            if active_hand_landmarks and is_mouse_control_active:
                # (Your FINGER_TAP_DOWN_Y_DELTA_THRESHOLD click logic - unchanged)
                current_index_tip_y = get_landmark(active_hand_landmarks, INDEX_TIP).y; current_middle_tip_y = get_landmark(active_hand_landmarks, MIDDLE_TIP).y
                if current_frame_timestamp_ms - last_left_click_time > CLICK_COOLDOWN_MS and (current_index_tip_y - prev_index_tip_y > FINGER_TAP_DOWN_Y_DELTA_THRESHOLD):
                    print("LEFT CLICK")
                    pyautogui.click(button='left'); last_left_click_time = current_frame_timestamp_ms
                    cv2.circle(display_frame_bgr, (int(get_landmark(active_hand_landmarks, INDEX_TIP).x * display_frame_bgr.shape[1]), int(current_index_tip_y * display_frame_bgr.shape[0])), 15, (0,255,0), -1)
                if current_frame_timestamp_ms - last_right_click_time > CLICK_COOLDOWN_MS and (current_middle_tip_y - prev_middle_tip_y > FINGER_TAP_DOWN_Y_DELTA_THRESHOLD):
                    print("RIGHT CLICK")
                    pyautogui.click(button='right'); last_right_click_time = current_frame_timestamp_ms
                    cv2.circle(display_frame_bgr, (int(get_landmark(active_hand_landmarks, MIDDLE_TIP).x * display_frame_bgr.shape[1]), int(current_middle_tip_y * display_frame_bgr.shape[0])), 15, (0,128,255), -1)
                prev_index_tip_y = current_index_tip_y; prev_middle_tip_y = current_middle_tip_y
            
            # --- UI Display ---
            new_frame_time = time.time()
            if (new_frame_time - prev_frame_time) > 0: fps = 1 / (new_frame_time - prev_frame_time)
            else: fps = 0; prev_frame_time = new_frame_time
            cv2.putText(display_frame_bgr, f"FPS: {fps:.2f}", (display_frame_bgr.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            mode_text = f"Mouse Ctrl: {'ON' if is_mouse_control_active else 'OFF'} (M)"; cv2.putText(display_frame_bgr, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if is_mouse_control_active else (0,0,255), 2)
            claw_text = f"Claw Active: {'YES' if is_dragon_claw_active_current_frame else 'NO'}"; cv2.putText(display_frame_bgr, claw_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0) if is_dragon_claw_active_current_frame else (128,128,128), 2)
            
            hand_in_roi_for_display = False # Default
            if active_hand_landmarks:
                palm_anchor_point = get_landmark(active_hand_landmarks, WRIST) # Recalculate for safety if not done above this block for UI
                hand_in_roi_for_display = (palm_anchor_point.x >= dynamic_roi_x_min and palm_anchor_point.x <= dynamic_roi_x_max and \
                                           palm_anchor_point.y >= dynamic_roi_y_min and palm_anchor_point.y <= dynamic_roi_y_max)

            cursor_status_text = "BLOCKED"
            if can_attempt_cursor_control and active_hand_landmarks and hand_in_roi_for_display: cursor_status_text = "ACTIVE_IN_ROI"
            elif can_attempt_cursor_control and active_hand_landmarks: cursor_status_text = "ACTIVE_OUT_ROI"
            elif can_attempt_cursor_control : cursor_status_text = "NO_HAND_BUT_ALLOWED" # Should not happen if claw needs hand
            cv2.putText(display_frame_bgr, f"Cursor: {cursor_status_text}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if update_cursor_this_frame else (0,0,255),2)

            # Draw Dynamic ROI
            roi_x_start_px = int(dynamic_roi_x_min * display_frame_bgr.shape[1]); roi_x_end_px = int(dynamic_roi_x_max * display_frame_bgr.shape[1])
            roi_y_start_px = int(dynamic_roi_y_min * display_frame_bgr.shape[0]); roi_y_end_px = int(dynamic_roi_y_max * display_frame_bgr.shape[0])
            cv2.rectangle(display_frame_bgr, (roi_x_start_px, roi_y_start_px), (roi_x_end_px, roi_y_end_px), (0, 255, 255), 2) # Yellow dynamic ROI
            cv2.circle(display_frame_bgr, (int(clamped_roi_center_x * display_frame_bgr.shape[1]), int(clamped_roi_center_y * display_frame_bgr.shape[0])), 5, (255,0,255), -1) # ROI Center
            if active_hand_landmarks: # Draw palm anchor
                cv2.circle(display_frame_bgr, (int(palm_anchor_point.x * display_frame_bgr.shape[1]), int(palm_anchor_point.y * display_frame_bgr.shape[0])), 7, (0,0,255), -1) # Wrist


            cv2.imshow(window_name, display_frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('m'): is_mouse_control_active = not is_mouse_control_active
        
        cap.release(); cv2.destroyAllWindows(); print("Program finished.")

if __name__ == '__main__':
    main()
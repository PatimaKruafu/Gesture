import mediapipe as mp
import cv2
import time

# --- Model Paths ---
hand_landmarker_path = 'tasks/hand_landmarker.task'
# gesture_recognizer_path = 'tasks/gesture_recognizer.task' # Comment out for now

# --- MediaPipe Task Setup ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
# GestureRecognizer = mp.tasks.vision.GestureRecognizer # Comment out
# GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions # Comment out
# GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult # Comment out
VisionRunningMode = mp.tasks.vision.RunningMode

# --- Global variable for result (minimal) ---
_latest_hand_result = None # Underscore to indicate internal use for this test

def hand_landmarker_callback_minimal(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global _latest_hand_result
    _latest_hand_result = result # Just store it, no processing
    # print(f"Hand callback at {timestamp_ms}") # Optional: for checking if callback fires

hand_options_minimal = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_landmarker_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1, # Simplify to 1 hand
    min_hand_detection_confidence=0.6, # Slightly higher confidence
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
    result_callback=hand_landmarker_callback_minimal)

def main_minimal_test():
    global _latest_hand_result

    with HandLandmarker.create_from_options(hand_options_minimal) as landmarker:
        cap = cv2.VideoCapture(0)
        # --- TRY REDUCING RESOLUTION DRASTICALLY ---
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320) # Even more drastic
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        window_name = 'Minimal Test'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        prev_frame_time = 0
        new_frame_time = 0

        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                break
            
            frame_bgr = cv2.flip(frame_bgr, 1) # Keep flip for consistency if testing cursor later

            frame_timestamp_ms = int(time.time() * 1000)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            landmarker.detect_async(mp_image, frame_timestamp_ms)
            
            # --- Display Logic (minimal) ---
            display_frame_bgr = frame_bgr.copy() 

            # --- FPS Calculation & Display ---
            new_frame_time = time.time()
            if (new_frame_time - prev_frame_time) > 0:
                fps = 1 / (new_frame_time - prev_frame_time)
            else:
                fps = 0 
            prev_frame_time = new_frame_time
            
            # Optional: Show if a hand was detected to confirm model is running
            if _latest_hand_result and _latest_hand_result.hand_landmarks:
                cv2.putText(display_frame_bgr, "Hand Detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            _latest_hand_result = None # Clear it for next frame detection check

            cv2.putText(display_frame_bgr, f"FPS: {fps:.2f}", (display_frame_bgr.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(window_name, display_frame_bgr)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Minimal test finished.")

if __name__ == '__main__':
    main_minimal_test()
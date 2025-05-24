import mediapipe as mp
import cv2

hand_landmarker_path = 'tasks/hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

mp_drawing = mp.solutions.drawing_utils

# Create a hand landmarker instance with the live stream mode:
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('hand landmarker result: {}'.format(result))

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_landmarker_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

#def findFinger



timestamp = 0

with HandLandmarker.create_from_options(options) as landmarker:
  cap = cv2.VideoCapture(1)
  cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
  cv2.resizeWindow('frame', 800, 800)

  while cap.isOpened():
    ret, frame = cap.read()
    timestamp += 1
    if not ret:
      print("Ignoring empty frame")
      break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    landmarker.detect_async(mp_image, timestamp)

    detection_result = HandLandmarkerOptions.result_callback
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if detection_result:
      image = mp_drawing.draw_landmarks(
        bgr_frame, detection_result, mp.hands.HAND_CONNECTIONS,
        #mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        #mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
      )

    cv2.imshow('frame', bgr_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()


# Use OpenCV’s VideoCapture to start capturing from the webcam.

# Create a loop to read the latest frame from the camera using VideoCapture#read()

# Convert the frame received from OpenCV to a MediaPipe’s Image object.
# mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)


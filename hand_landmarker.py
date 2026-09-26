import cv2
import pprint

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)

# STEP 3: Load the input image.
image = mp.Image.create_from_file("image.jpg")
with vision.HandLandmarker.create_from_options(options) as detector:
    # STEP 4: Detect hand landmarks from the input image.
    detection_result = detector.detect(image)

img = cv2.imread("image.jpg")
h, w = img.shape[:2]

for handedness, landmarks in zip(
    detection_result.handedness, detection_result.hand_landmarks
):
    x_y_list = []
    for coordinate in landmarks:
        format_x, format_y = int(coordinate.x * w), int(coordinate.y * h)
        cv2.circle(img, (format_x, format_y), 5, (0, 0, 255), thickness=-1)
        x_y_list.append((format_x, format_y))

    min_x, min_y = x_y_list[0]
    for x, y in x_y_list:
        min_x = min(min_x, x)
        min_y = min(min_y, y)

    cv2.putText(
        img,
        text=handedness[0].category_name,
        org=(max(min_x - 20, 0), max(min_y - 20, 30)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.2,
        color=(0, 0, 255),
        thickness=3,
    )

    for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(
            img,
            x_y_list[connection.start],
            x_y_list[connection.end],
            (0, 255, 0),
            thickness=2,
        )

cv2.imwrite("output.jpg", img)

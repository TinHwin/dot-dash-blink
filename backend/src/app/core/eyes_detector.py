import cv2 as cv
import mediapipe as mp

from pathlib import Path

from mediapipe.tasks.python.vision import FaceLandmarksConnections

MODEL_PATH = (
    Path(__file__).resolve().parents[3]
    / "models"
    / "face_landmarker.task"
)

BaseOptions = mp.tasks.BaseOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker

FaceLandmarkerOptions = (
    mp.tasks.vision.FaceLandmarkerOptions
)

VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options = BaseOptions(
        model_asset_path = str(MODEL_PATH)
    ),
    running_mode = VisionRunningMode.VIDEO,
    num_faces = 1,
    output_face_blendshapes = True
)

landmarker = FaceLandmarker.create_from_options(options)

class EyesDetector:
    def __init__(self, camera):
        self.camera = camera
        self.landmarks = []

    def get_landmarks(self, bgr_frame):
        # Convert BGR to RGB
        rgb_frame = cv.cvtColor(
            bgr_frame,
            cv.COLOR_BGR2RGB
        )

        # Convert to MediaPipe image
        mp_image = mp.Image(
            image_format = mp.ImageFormat.SRGB,
            data = rgb_frame
        )

        result = landmarker.detect_for_video(
            mp_image,
            self.camera.get_timestamp_ms()
        )

        if result.face_landmarks:
            for landmarks in result.face_landmarks:
                height, width, _ = bgr_frame.shape

                for connections in (FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE, FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE):
                    for connection in connections:
                        start = landmarks[connection.start]
                        end = landmarks[connection.end]

                        x1 = int(start.x * width)
                        y1 = int(start.y * height)

                        x2 = int(end.x * width)
                        y2 = int(end.y * height)

                        cv.line(
                            bgr_frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            1
                        )

                # Draw facial landmark points
                for landmark in landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    cv.circle(
                        bgr_frame,
                        (x, y),
                        1,
                        (0, 255, 0),
                        -1
                    )

        self.landmarks = result.face_landmarks

        if result.face_blendshapes:
            blendshapes = result.face_blendshapes[0]

            top_blendshapes = sorted(
                blendshapes,
                key = lambda item: item.score,
                reverse = True
            )[:5]

            y_position = 30

            for category in top_blendshapes:
                text = (
                    f"{category.category_name}: "
                    f"{category.score:.2f}"
                )

                cv.putText(
                    bgr_frame,
                    text,
                    (20, y_position),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1
                )

                y_position += 25

        return self.landmarks
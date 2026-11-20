import cv2 as cv

from camera import Camera
from eyes_detector import EyesDetector
from blinks_analyzer import BlinksAnalyzer

def main():
    camera = Camera()
    eyes_detector = EyesDetector(camera)
    blinks_analyzer = BlinksAnalyzer()

    try:
        while True:
            frame = camera.get_frame()

            if frame is None:
                break

            landmarks = eyes_detector.get_landmarks(frame)

            is_blinking = blinks_analyzer.is_blinking(landmarks)
            
            cv.imshow(
                "MediaPipe Facial Landmarks",
                frame
            )

            if cv.waitKey(1) & 0xFF == ord("q"):
                break 
    finally:
        camera.release()

if __name__ == "__main__":
    main()
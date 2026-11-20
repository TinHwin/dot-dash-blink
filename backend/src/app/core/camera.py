import cv2 as cv

class Camera:
    def __init__(self, device_index = 0):
        self.cap = cv.VideoCapture(device_index)

        self.frame_index = 0

        self.fps = self.cap.get(cv.CAP_PROP_FPS) or 30.0

        self.timestamp_ms = round(
            (self.frame_index * 1000.0) / self.fps
        )

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

    def get_frame(self):
        ret, frame = self.cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")

            return None

        self.increment_timestamp_ms()

        self.increment_frame_index()

        return frame

    def get_frame_index(self):
        return self.frame_index

    def increment_frame_index(self):
        self.frame_index += 1

    def get_fps(self):
        return self.fps

    def get_timestamp_ms(self):
        return self.timestamp_ms

    def increment_timestamp_ms(self):
        self.timestamp_ms = round(
            (self.frame_index * 1000.0) / self.fps
        )

    def release(self):
        self.cap.release()
        cv.destroyAllWindows()
        
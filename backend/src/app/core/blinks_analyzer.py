class BlinksAnalyzer:
    def get_eye_aspect_ratio(self, landmarks, side):
        if side == "left":
            upper = landmarks[0][159]
            lower = landmarks[0][145]
            outer = landmarks[0][33]
            inner = landmarks[0][133]

            vertical = abs(upper.y - lower.y)
            horizontal = abs(outer.x - inner.x)

            return vertical / horizontal
        elif side == "right":
            upper = landmarks[0][386]
            lower = landmarks[0][374]
            outer = landmarks[0][263]
            inner = landmarks[0][362]

            vertical = abs(upper.y - lower.y)
            horizontal = abs(outer.x - inner.x)

            return vertical / horizontal

    def is_blinking(self, landmarks):
        if not landmarks:
            return False

        leftEyeAspectRatio = self.get_eye_aspect_ratio(landmarks, "left")
        rightEyeAspectRatio = self.get_eye_aspect_ratio(landmarks, "right")

        if leftEyeAspectRatio < 0.2 and rightEyeAspectRatio < 0.2:
            return True
        else:
            return False
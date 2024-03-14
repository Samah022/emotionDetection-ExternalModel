from ..criteriaDetectionBehavior.detectionBehavior import DetectionBehavior

class UniformDetection(DetectionBehavior):
    
    def detect(self, frame, cameraID):
        return True
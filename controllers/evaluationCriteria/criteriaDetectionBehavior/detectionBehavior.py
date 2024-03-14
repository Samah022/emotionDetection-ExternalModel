from ..detectionModelAdaptor.detectionModelAdaptor import DetectionModelAdaptor

class DetectionBehavior:
    # def __init__(self, behavior_name):
    #     self.behavior_name = behavior_name

    # def __str__(self):
    # return f"DetectionBehavior(behavior_name={self.behavior_name})"

    def __init__(self, criteria_id, criteria_name, model):
        self.__criteria_id = criteria_id
        self.__criteria_name = criteria_name
        self.__model = DetectionModelAdaptor()
    
    def detect(self, frame, cameraID):
        return "test"
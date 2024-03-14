from ..detectionModelAdaptor.detectionModelAdaptor import DetectionModelAdaptor
from ...servicesHandler.servicesHandler import ServicesHandler

class FaceModel(DetectionModelAdaptor):
    def __init__(self):
        self.__services_handler = ServicesHandler()
    
    def predict(self, frame):
        faces = self.__services_handler.handle_face_predict_request(frame)
        # print(f"face model face result: {faces}")
        return faces
from ..detectionModelAdaptor.detectionModelAdaptor import DetectionModelAdaptor
from ...servicesHandler.servicesHandler import ServicesHandler
import cv2
from fer import FER


class EmotionModel(DetectionModelAdaptor):

    def __init__(self):
        self.__services_handler = ServicesHandler()

    def predict(self, camera_url):
        face_id, emotion_result, emotion_score = self.__services_handler.handle_emotion_predict(
            camera_url)
        print(f"emotion model emotion result: {face_id, emotion_result, emotion_score}")
        return face_id, emotion_result, emotion_score

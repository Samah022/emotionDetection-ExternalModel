from models.DB_management.DBFaced import DBFacade
from ..criteriaDetectionBehavior.detectionBehavior import DetectionBehavior
from .faceDetection import FaceDetection
from ..emotionModel.emotionModel import EmotionModel
from collections import defaultdict
from inference.core.interfaces.camera.entities import VideoFrame
from inference import InferencePipeline
from datetime import datetime


class EmotionsDetection(DetectionBehavior):
    def __init__(self):
        self.__db_facade = DBFacade()
        self.__face_detector = FaceDetection()
        self.__emotion_predictor = EmotionModel()
        self.DB_result = []

    def my_custom_sink(self, predictions: dict, video_frame: VideoFrame):  # Added self parameter
        # get the text labels for each prediction
        labels = [p["class"] for p in predictions["predictions"]]
        amount = len(labels)  # Assuming amount is the count of detected emotions
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print({ "emotion": labels, "amount": amount, "time": current_time})

    def detect(self, frames, cameraID):
        pipeline = InferencePipeline.init(
            model_id="emotion-detection-cwq4g/1",
            api_key="CL44RJt0AHwiczZPxMLN",
            video_reference="rtsp://192.168.214.114:8080/h264_ulaw.sdp",
            on_prediction=self.my_custom_sink,  # Fixed method reference
        )
        pipeline.start()
        pipeline.join()



    def __guess_emotion_with_weights(self, emotion_results):
        emotion_stats = defaultdict(lambda: {'total_score': 0, 'count': 0})

        for result in emotion_results:
            emotion = result['emotion']
            score = result['score']
            if emotion is not None and score is not None:
                emotion_stats[emotion]['total_score'] += score
                emotion_stats[emotion]['count'] += 1

        weighted_emotions = {}
        for emotion, stats in emotion_stats.items():
            if stats['count'] > 0:
                weighted_score = stats['total_score'] / stats['count']  # Weighted by the frequency
                weighted_emotions[emotion] = weighted_score

        guessed_emotion = max(weighted_emotions, key=weighted_emotions.get) if weighted_emotions else None
        return guessed_emotion

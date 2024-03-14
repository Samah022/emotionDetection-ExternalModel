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

    # def detect(self, frames, camera_ID, camera_url):

    #     pipeline = InferencePipeline.init(
    #         model_id="emotion-detection-cwq4g/1",
    #         api_key="CL44RJt0AHwiczZPxMLN",
    #         video_reference="rtsp://192.168.214.114:8080/h264_ulaw.sdp",
    #         on_prediction=lambda predictions, video_frame: self.my_custom_sink(
    #             predictions, video_frame, camera_ID),  # Fixed method reference
    #     )
    #     pipeline.start()
    #     pipeline.join()

    def detect(self,cameraID,camera_url):
        
        frames_result = []
        combined_result = []
        DB_result = []
     
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        face_id, emotion_result, emotion_score = self.__emotion_predictor.predict(camera_url)

        frames_result.append( {"id": face_id, "emotion": emotion_result, "score": emotion_score})
        print({"id": face_id, "emotion": emotion_result, "score": emotion_score})

        # {'emotion': ['happy', 'sad'], 'score': [0.8, 0.6]}

        # Group frames_result by ID
        grouped_results = {}
        for frame_result in frames_result:
            id = frame_result["id"]
            if id not in grouped_results:
                grouped_results[id] = []
            grouped_results[id].append(frame_result)

        # Process grouped results
        for id, results in grouped_results.items():
            combined_result.append({"id": id, "emotion": self.__guess_emotion_with_weights(results)})

        emotion_count = []
        for entry in combined_result:
            emotion = entry["emotion"]
            found = False
            for item in emotion_count:
                if item["emotion"] == emotion:
                    item["amount"] += 1
                    found = True
                    break
            if not found:
                emotion_count.append({"emotion": emotion, "amount": 1})

        for entry in emotion_count:
            emotion = entry["emotion"]
            amount = entry["amount"]
            DB_result.append({"cameraID": cameraID, "emotion": emotion,
                             "amount": amount, "time": current_time})

        for result in DB_result:
            self.__db_facade.set_emotion_data(
                result["time"], result["emotion"], result["amount"], result["cameraID"])
        print("result in DB", DB_result)

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
                weighted_score = stats['total_score'] / \
                    stats['count']  # Weighted by the frequency
                weighted_emotions[emotion] = weighted_score

        guessed_emotion = max(
            weighted_emotions, key=weighted_emotions.get) if weighted_emotions else None
        return guessed_emotion

from models.DB_management.DBFaced import DBFacade
from ..criteriaDetectionBehavior.detectionBehavior import DetectionBehavior
from .faceDetection import FaceDetection
from ..emotionModel.emotionModel import EmotionModel
from collections import defaultdict
import sqlite3
import time
from datetime import datetime
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame

# import DB to save max frames_result result 

class EmotionsDetection(DetectionBehavior):
    def __init__(self):
        self.__db_facade = DBFacade()
        self.__face_detector = FaceDetection()
        self.__emotion_predictor = EmotionModel()

    def detect(self, frames, cameraID):
        frames_result = []
        combined_result = []
        DB_result = []

        for  frame in frames:
            cropped_faces = self.__face_detector.detect(frame)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for id, face in enumerate(cropped_faces):
                emotion_result, emotion_score = self.__emotion_predictor.predict(face)
                frames_result.append({"id": id, "emotion":emotion_result,"score": emotion_score})
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
            DB_result.append({"cameraID": cameraID, "emotion": emotion, "amount": amount, "time": current_time})

        for result in DB_result:
            self.__db_facade.set_emotion_data(result["time"],result["emotion"], result["amount"], result["cameraID"] )
        print("result", DB_result)
        

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
 
    # def __guess_emotion_with_weights(self, emotion_results):
    #     emotion_stats = defaultdict(lambda: {'total_score': 0, 'count': 0})
        
    #     for result in emotion_results:
    #         emotion = result['emotion']
    #         score = result['score']
    #         if emotion is not None and score is not None:
    #             emotion_stats[emotion]['total_score'] += score
    #             emotion_stats[emotion]['count'] += 1
        
    #     weighted_emotions = {}
    #     for emotion, stats in emotion_stats.items():
    #         if stats['count'] > 0:
    #             weighted_score = stats['total_score'] / stats['count']  # Weighted by the frequency
    #             weighted_emotions[emotion] = weighted_score
        
    #     guessed_emotion = max(weighted_emotions, key=weighted_emotions.get) if weighted_emotions else None
    #     return guessed_emotion

    # def __guess_emotion_with_weights(self, emotion_results):
    #     emotion_stats = defaultdict(lambda: {'total_score': 0, 'count': 0})
        
    #     for result in emotion_results:
    #         if isinstance(result, set):
    #             if result != {None}:  # Exclude frames with no detected emotions
    #                 if isinstance(list(result)[0], tuple):  # Check if it's a tuple
    #                     emotion, score = list(result)[0]  # Unpack the tuple
    #                     emotion_stats[emotion]['total_score'] += score
    #                     emotion_stats[emotion]['count'] += 1
    #                 else: 
    #                     score, emotion = result.pop(), result.pop()  
    #                     emotion_stats[emotion]['total_score'] += score
    #                     emotion_stats[emotion]['count'] += 1
        
    #     weighted_emotions = {}
    #     for emotion, stats in emotion_stats.items():
    #         if stats['count'] > 0:
    #             weighted_score = stats['total_score'] / stats['count']  # Weighted by the frequency
    #             weighted_emotions[emotion] = weighted_score
        
    #     guessed_emotion = max(weighted_emotions, key=weighted_emotions.get) if weighted_emotions else None
    #     return guessed_emotion

    
    


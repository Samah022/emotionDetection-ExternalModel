from fer import FER
import cv2
from inference.core.interfaces.camera.entities import VideoFrame
from inference import InferencePipeline
from datetime import datetime

class ServicesHandler:

    
    # def handle_emotion_predict(self, frame):
    #     emotion_detector = FER()
    #     emotions = emotion_detector.detect_emotions(frame)
    #     if emotions:
    #         emotion = emotions[0]['emotions']
    #         max_emotion = max(emotion, key=emotion.get)
    #         max_score = emotion[max_emotion]
    #         return max_emotion, max_score
    #     else:
    #         return None, None
        
    # def my_custom_sink(self, predictions: dict, video_frame: VideoFrame):  # Added camera_ID parameter
    #     labels_confidence = [(p["class"], p["confidence"]) for p in predictions["predictions"]]
    #     print(labels_confidence)
    #     return labels_confidence
    #[('happy', 0.8), ('sad', 0.6), ('angry', 0.7)]

    def my_custom_sink(self,predictions: dict, video_frame: VideoFrame):  
            labels_confidence = [(p["class"], p["confidence"]) for p in predictions["predictions"]]
            if labels_confidence:
                for id, label_confidence in enumerate(labels_confidence):
                    emotion_result = label_confidence[0]
                    emotion_score = label_confidence[1]
                    face_id = id
                    
                    # emotion_result = [label for label, score in labels_confidence]
                    # emotion_score = [score for label, score in labels_confidence]
                    # result = {"emotion": emotion_result, "score": emotion_score}

                    # print("service handler",face_id, emotion_result, emotion_score)
                    return face_id, emotion_result, emotion_score
            
    def handle_emotion_predict(self, camera_url):
        pipeline = InferencePipeline.init(
            model_id="emotion-detection-cwq4g/1",
            api_key="CL44RJt0AHwiczZPxMLN",
            video_reference= camera_url,
            on_prediction=lambda predictions, video_frame: self.my_custom_sink(predictions, video_frame),  # Fixed method reference
        )
        pipeline.start()
        pipeline.join()

        

    def handle_face_predict_request(self, frame):
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            return faces
        
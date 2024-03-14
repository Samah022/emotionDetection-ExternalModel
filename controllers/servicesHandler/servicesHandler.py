from threading import Thread
from queue import Queue

from fer import FER
import cv2
from inference.core.interfaces.camera.entities import VideoFrame
from inference import InferencePipeline
from datetime import datetime


class ServicesHandler:

    def handle_emotion_predict(self, camera_url):
        # Create a thread-safe queue for storing predictions
        predictions_queue = Queue()

        # Define a function for printing from the queue
        def print_predictions():
            while True:
                prediction = predictions_queue.get()  # Block until an item is available
                if prediction is None:
                    break  # Stop printing when None is received (optional)
                face_id, emotion_result, emotion_score = prediction
                print(f"Face ID: {face_id}, Emotion: {emotion_result}, Score: {emotion_score}")

        # Start a printing thread
        print_thread = Thread(target=print_predictions, daemon=True)
        print_thread.start()

        pipeline = InferencePipeline.init(
            model_id="emotion-detection-cwq4g/1",
            api_key="CL44RJt0AHwiczZPxMLN",
            video_reference=camera_url,
            on_prediction=lambda predictions, video_frame: self.my_custom_sink(predictions, video_frame, predictions_queue)
        )

        pipeline.start()
        pipeline.join()

        # Send a None signal to the printing thread to stop (optional)
        predictions_queue.put(None)

        # Wait for the printing thread to finish (optional)
        print_thread.join()

        return None  # Optional: return if needed

    def my_custom_sink(self, predictions: dict, video_frame: VideoFrame, predictions_queue: Queue):
        labels_confidence = [(p["class"], p["confidence"]) for p in predictions["predictions"]]
        if labels_confidence:
            for id, label_confidence in enumerate(labels_confidence):
                face_id, emotion_result, emotion_score = id, label_confidence[0], label_confidence[1]
                predictions_queue.put((face_id, emotion_result, emotion_score))  # Add prediction to queue
        else:
            predictions_queue.put(None)  # Add None if no emotions detected

    def handle_face_predict_request(self, frame):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces

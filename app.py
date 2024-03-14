# # Import the InferencePipeline object
from inference import InferencePipeline
# Import the built in render_boxes sink for visualizing results
from inference.core.interfaces.stream.sinks import render_boxes
from roboflow import Roboflow
import supervision as sv
import cv2

# initialize a pipeline object
# pipeline = InferencePipeline.init(
#     model_id="emotion-detection-cwq4g/1", # Roboflow model to use
#     api_key="CL44RJt0AHwiczZPxMLN",
#     video_reference=0 ,# Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
#     on_prediction=render_boxes,  # Function to run after each prediction
# )
# print(pipeline)


# pipeline.start()
# pipeline.join()

# # Import the InferencePipeline object
# from inference import InferencePipeline
# from roboflow import Roboflow
# import supervision as sv
# import cv2

# # Initialize a pipeline object
# pipeline = InferencePipeline.init(
#     model_id="emotion-detection-cwq4g/1",  # Roboflow model to use
#     api_key="CL44RJt0AHwiczZPxMLN",
#     video_reference= 0,  # Path to video, device id (int, usually 0 for built-in webcams), or RTSP stream URL
#     on_prediction=lambda frame_id, prediction: print(frame_id, prediction),  # Print the frame_id and prediction
# )

# pipeline.start()
# pipeline.join()

#3
# from inference import InferencePipeline
# from roboflow import Roboflow
# import supervision as sv
# import cv2
# import time

# # Define a function to print the prediction result after a delay of 30 seconds
# def print_prediction_after_delay(frame_id, prediction):
#     time.sleep(30)  # Delay for 30 seconds
#     print("Frame ID:", frame_id)
#     print("Prediction:", prediction)

# # Initialize a pipeline object
# pipeline = InferencePipeline.init(
#     model_id="emotion-detection-cwq4g/1",  # Roboflow model to use
#     api_key="CL44RJt0AHwiczZPxMLN",
#     video_reference=0,  # Path to video, device id (int, usually 0 for built-in webcams), or RTSP stream URL
#     on_prediction=print_prediction_after_delay,  # Call the function to print prediction after a delay
# )

# pipeline.start()
# pipeline.join()

# from inference import InferencePipeline
# from inference.core.interfaces.camera.entities import VideoFrame

# def my_custom_sink(predictions: dict, video_frame: VideoFrame):
#     # get the text labels for each prediction
#     labels = [p["class"] for p in predictions["predictions"]]
#     # print the class names to the console
#     print("Detected classes:", labels)

# pipeline = InferencePipeline.init(
#     model_id="emotion-detection-cwq4g/1",
#     api_key="CL44RJt0AHwiczZPxMLN",
#     video_reference="rtsp://192.168.214.114:8080/h264_ulaw.sdp",
#     on_prediction=my_custom_sink,
# )

# pipeline.start()
# pipeline.join()

from inference.core.interfaces.camera.entities import VideoFrame
from inference import InferencePipeline

def my_custom_sink(predictions: dict, video_frame: VideoFrame):  
        labels_confidence = [(p["class"], p["confidence"]) for p in predictions["predictions"]]
        if labels_confidence:
            for id, label_confidence in enumerate(labels_confidence):
                emotion_result = label_confidence[0]
                emotion_score = label_confidence[1]
                face_id = id
                
                # emotion_result = [label for label, score in labels_confidence]
                # emotion_score = [score for label, score in labels_confidence]
                # result = {"emotion": emotion_result, "score": emotion_score}

                print("service handler",face_id, emotion_result, emotion_score)
                # return face_id, emotion_result, emotion_score
            

pipeline = InferencePipeline.init(
            model_id="emotion-detection-cwq4g/1",
            api_key="CL44RJt0AHwiczZPxMLN",
            video_reference= "rtsp://192.168.70.114:8080/h264_ulaw.sdp",
            on_prediction= lambda predictions, video_frame: my_custom_sink(predictions, video_frame),  # Fixed method reference
            # on_prediction=render_boxes, 
        )
pipeline.start()
pipeline.join()

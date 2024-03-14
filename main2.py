# # libaray
# import cv2
# from fer import FER

# # Load the pre-trained emotion detection model
# emotion_detector = FER()

# # Load the pre-trained face detection model
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Capture video from the camera
# cap = cv2.VideoCapture(0)

# while True:
#     # Read a frame from the video stream
#     ret, frame = cap.read()

#     # Convert the frame to grayscale for face detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the grayscale frame
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#     print(faces)

#     # Process each detected face
#     for (x, y, w, h) in faces:
#         # Draw a rectangle around the face
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#         # Crop the face region from the frame
#         face_roi = frame[y:y + h, x:x + w]

#         # Detect emotions in the face region
#         emotions = emotion_detector.detect_emotions(face_roi)

#         if emotions:
#             # Get the emotion with the highest probability
#             emotion = emotions[0]['emotions']
#             max_emotion = max(emotion, key=emotion.get)

#             # Print the highest emotion result on the screen
#             text = f"Emotion: {max_emotion}"
#             cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     # Display the frame
#     cv2.imshow('Frame', frame)

#     # Break the loop if the 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture and close all windows
# cap.release()
# cv2.destroyAllWindows()

# from roboflow import Roboflow
# rf = Roboflow(api_key="rT2aUciouxoBVB6u4B49")
# project = rf.workspace().project("emociones")
# model = project.version(4).model

# # infer on a local image
# print(model.predict("your_image.jpg", confidence=40, overlap=30).json())

# ##############################################################################################
# # outsourse

# # import cv2
# # from roboflow import Roboflow

# # # Initialize Roboflow API
# # rf = Roboflow(api_key="rT2aUciouxoBVB6u4B49")
# # project = rf.workspace().project("emociones")
# # model = project.version(4).model

# # # Initialize camera capture
# # cap = cv2.VideoCapture(0)

# # while True:
# #     # Read frame from camera
# #     ret, frame = cap.read()

# #     # Perform inference on the frame
# #     result = model.predict(frame, confidence=40, overlap=30).json()
# #     print(result)

# #     # Draw rectangle on the frame
# #     if "predictions" in result:
# #         predictions = result["predictions"]
# #         for obj in predictions:
# #             x, y, w, h = int(obj["x"]), int(obj["y"]), int(obj["width"]), int(obj["height"])
# #             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
# #             cv2.putText(frame, obj["class"], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# #     # Display the frame
# #     cv2.imshow('Camera', frame)

# #     # Break the loop if 'q' is pressed
# #     if cv2.waitKey(1) == ord('q'):
# #         break

# # # Release the camera and close windows
# # cap.release()
# # cv2.destroyAllWindows()
# ###########################################################3
# # outsourse

# # import cv2
# # from roboflow import Roboflow

# # # Initialize Roboflow API
# # rf = Roboflow(api_key="rT2aUciouxoBVB6u4B49")
# # project = rf.workspace().project("emotion-detection-cwq4g")
# # model = project.version(1).model

# # # Initialize camera capture
# # cap = cv2.VideoCapture(0)

# # while True:
# #     # Read frame from camera
# #     ret, frame = cap.read()

# #     # Perform inference on the frame
# #     result = model.predict(frame, confidence=40, overlap=30).json()
# #     print(result)

# #     # Draw rectangle on the frame
# #     if "predictions" in result:
# #         predictions = result["predictions"]
# #         for obj in predictions:
# #             x, y, w, h = int(obj["x"]), int(obj["y"]), int(obj["width"]), int(obj["height"])
# #             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
# #             cv2.putText(frame, obj["class"], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# #     # Display the frame
# #     cv2.imshow('Camera', frame)

# #     # Break the loop if 'q' is pressed
# #     if cv2.waitKey(1) == ord('q'):
# #         break

# # # Release the camera and close windows
# # cap.release()
# # cv2.destroyAllWindows()

# #####################################################
# # outsourse
# #not work

# # import cv2
# # from roboflow import Roboflow

# # # Initialize Roboflow API
# # rf = Roboflow(api_key="rT2aUciouxoBVB6u4B49")
# # project = rf.workspace().project("face-frown")
# # model = project.version(3).model

# # # Initialize camera capture
# # cap = cv2.VideoCapture(0)

# # while True:
# #     # Read frame from camera
# #     ret, frame = cap.read()

# #     # Perform inference on the frame
# #     result = model.predict(frame, confidence=40, overlap=30).json()
# #     print(result)

# #     # Draw rectangle on the frame
# #     if "predictions" in result:
# #         predictions = result["predictions"]
# #         for obj in predictions:
# #             x, y, w, h = int(obj["x"]), int(obj["y"]), int(obj["width"]), int(obj["height"])
# #             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
# #             cv2.putText(frame, obj["class"], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# #     # Display the frame
# #     cv2.imshow('Camera', frame)

# #     # Break the loop if 'q' is pressed
# #     if cv2.waitKey(1) == ord('q'):
# #         break

# # # Release the camera and close windows
# # cap.release()
# # cv2.destroyAllWindows()

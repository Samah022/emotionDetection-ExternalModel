from ..faceModel.faceModel import FaceModel

class FaceDetection:
    def __init__(self):
        self.__face_model = FaceModel()

    def detect(self, frame):
        cropped_faces = []
        faces = self.__face_model.predict(frame)
        for (x, y, w, h) in faces: 
            face_location = frame[y:y + h, x:x + w]
            cropped_faces.append(face_location)
        # print("face detect",cropped_faces )
        return cropped_faces

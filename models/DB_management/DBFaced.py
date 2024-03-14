from .CameraDB import CameraDB
from .emotionDB import EmotionDB

cameraDB = CameraDB()
emotion_db = EmotionDB()


class DBFacade:

    
    def get_all_cameras (self):
        return cameraDB.getAllCameras()
    
    def set_emotion_data(self, timestamp, emotion_type, amount, camera_id):
            print("facade file")
            return emotion_db.set_emotion_data(timestamp, emotion_type, amount, camera_id)
    

    



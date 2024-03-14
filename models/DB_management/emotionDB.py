from ..evaluationCriteriaManagement.emotionEntity import Emotion
import sqlite3

class EmotionDB:

    def set_emotion_data(self, timestamp, emotion_type, amount, camera_id):
        try:
            emotion_obj = Emotion()
            
            conn = sqlite3.connect('monitoring-and-evaluation.db')  
            cr = conn.cursor()

            emotion_obj.Timestamp = timestamp
            emotion_obj.Type = emotion_type
            emotion_obj.Amount = amount
            emotion_obj.Camera_ID = camera_id

            sql_query = """ 
                    INSERT INTO Emotion (Timestamp, Type, Amount, Camera_ID)
                    VALUES (?, ?, ?, ?);
                """
            print("wmotion file 1")
            cr.execute(sql_query, (timestamp, emotion_type, amount, camera_id))
            print("wmotion file 2")
            conn.commit()

        except Exception as e:
            print("Error connecting to database or retrieving data: ", e)
            return None

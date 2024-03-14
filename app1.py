from models.DB_management.DBFaced import DBFacade

db_facade = DBFacade()

timestamp = "2024-03-05 12:00:00"
emotion_type = "Sad"
amount = 2
camera_id = 1

db_facade.set_emotion_data(timestamp, emotion_type, amount, camera_id)
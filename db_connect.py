# Method-1: use sqllite3
import sqlite3

db = sqlite3.connect("monitoring-and-evaluation.db") 
cr = db.cursor()
print("ffff")
# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker

# engine = create_engine("sqlite:///monitoring-and-evaluation.db")
# Session = sessionmaker(bind=engine)
# session = Session()

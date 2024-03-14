import asyncio
# from controllers.camera.cameraBoard import CameraBoard
from controllers.camera.camera import Camera


async def process_cameras():
    # board = CameraBoard()
    # result = board.get_all_cameras()
    # result = [Camera(1,"test1","rtsp://192.168.253.114:8080/h264_ulaw.sdp",["emotion"]),Camera(2,"test2",0,["emotion"])]
    
    result = [Camera(1, "test1", "rtsp://192.168.70.114:8080/h264_ulaw.sdp", ["emotion"])]
    tasks = [camera.analysis_frame()
            for camera in result]
    await asyncio.gather(*tasks)

asyncio.run(process_cameras())

# p1=threading.Thread(target=generate_frame)
# p2 = threading.Thread(target=analysis_frame)
# p1.start()
# p2.start()


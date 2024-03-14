from ...models.DB_management.DBFaced import DBFacade
from .camera import Camera

class CameraBoard:
    def __init__(self):
        self.__camera_list = []
        self.__DBFacade= DBFacade()


    def get_all_cameras(self):
        result = self.__DBFacade.get_all_cameras()
        for camera in result:
            fetched_camera = Camera(ID=camera["id"],camera_name=camera["name"],camera_URL=camera["link"], evaluation_criteria=camera["criteria"])
            self.__camera_list.append(fetched_camera)
        return self.__camera_list
        

    




                    


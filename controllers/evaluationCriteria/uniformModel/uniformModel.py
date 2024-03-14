from ..detectionModelAdaptor.detectionModelAdaptor import DetectionModelAdaptor

class UniformModel(DetectionModelAdaptor):

    def __init__(self, apiKey, projectName, version):
        self.__apiKey = apiKey
        self.__projectName = projectName
        self.__version = version

    def predict(self, frame):
        return "test"

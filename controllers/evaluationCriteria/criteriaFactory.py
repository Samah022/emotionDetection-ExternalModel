from .dressCodeDetection.uniformDetection import UniformDetection
from .emotionDetection.emotionsDetection import EmotionsDetection

class CriteriaFactory:
    
    def createCriteria(self,criteriaName):
        if criteriaName == "emotion":
            return EmotionsDetection()

        elif criteriaName == "uniform":
            return UniformDetection()

import cv2
import numpy as np

class PreprocessingFrame:
    def __init__(self, resize_scale):
        self.__resize_scale = self.set_resize_scale(1)

    """
    Returns the resize scale value used for preprocessing.

    Returns:
        float: The resize scale value.
    """
    def get_resize_scale(self):
        return self.__resize_scale
    

    """
    Sets the resize scale value used for preprocessing.

    Args:
        resize_scale (int or float): The new resize scale value.

    Raises:
        TypeError: If the resizeScale is not an int or float.

    Returns:
        (int or float): Resize scale value if set correctly.
    """
    def set_resize_scale(self, resize_scale):
        try:
            if isinstance(resize_scale, (int, float)) and resize_scale > 0:
                self.__resize_scale = resize_scale
                return resize_scale
            else:
                raise ValueError("Invalid resizeScale value. Expected a positive and non-zero int or float. resizeScale has been set to 1.")
        except Exception as e:
            print(e)
            return ""
        
    """
    Optimizes the given frame by performing preprocessing steps.
    Preprocessing Steps:
    1. Resize the frame
    2. Convert the frame to grayscale
    3. Enhance the contrast of the frame

    Args:
        frame (int []): The frame to be optimized.

    Raises:
        TypeError: If the frame is not of the expected type.

    Returns:
        int []: The optimized frame as a NumPy array.

    """    
    def optimize_frame(self, frame):
        try:
            if not isinstance(frame, np.ndarray):
                raise TypeError("Invalid frame type. Expected numpy.ndarray.")
            
            resized_frame = self.__resize(frame)

            grayscale_frame = self.__convert_to_grayscale(resized_frame)

            # contrast_enhanced_frame = self.__enhance_contrast(grayscale_frame)

            return grayscale_frame
        except Exception as e:
            print(e)
            return None
        
    """
    Converts the given frame to grayscale.

    Args:
        frame (int [] ): The frame to be converted.

    Raises:
        TypeError: If the frame is not of the expected type.
        ValueError: If the frame is not in RGB color format.

    Returns:
        int [] : The grayscale frame as a NumPy array.
    """    
    
    def __convert_to_grayscale(self, frame):
            try:
                if not isinstance(frame, np.ndarray):
                    raise TypeError("Invalid frame type. Expected numpy.ndarray.")

                if frame.shape[-1] != 3 or frame.ndim != 3:
                    raise ValueError("Invalid frame format. Expected RGB color format.")

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                return gray_frame
            except Exception as e:
                print(e)
                return None
            
    """
    Enhances the contrast of the given frame.

    Args:
        frame (int []): The frame to have its contrast enhanced.

    Raises:
        TypeError: If the frame is not of the expected type.
        ValueError: If the frame is not in grayscale format.

    Returns:
        int []: The contrast-enhanced frame as a NumPy array.
    """
    
    def __enhance_contrast(self, frame):
        try:
            if not isinstance(frame, np.ndarray):
                raise TypeError("Invalid frame type. Expected numpy.ndarray.")

            if len(frame.shape) != 2:
                raise ValueError("Invalid frame format. Expected grayscale image.")

            equalized_frame = cv2.equalizeHist(frame)
            return equalized_frame
        except Exception as e:
            print(e)
            return None
        
    """
    Resizes the given frame.

    Args:
        frame (int [] ): The frame to be resized.

    Raises:
        TypeError: If the frame is not of the expected type.
        ValueError: If the frame is None.

    Returns:
        int [] : The resized frame as a NumPy array.
    """
    
    def __resize(self, frame):
        try:
            if frame is None:
                raise ValueError("Invalid frame: None")

            if not isinstance(frame, np.ndarray):
                raise TypeError("Invalid frame type. Expected numpy.ndarray.")

            resized_frame = cv2.resize(frame, None, fx=self.__resize_scale, fy=self.__resize_scale)
            return resized_frame
        except Exception as e:
            print(e)
            return None


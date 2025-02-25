from abc import ABC, abstractmethod

class ImageProcessingBase(ABC):
    @abstractmethod
    def process_image(self, image):
        """
        Defines a blueprint for the process image method.
        Takes an PIL RGB image as input and return labelsor others as list
        """
        pass
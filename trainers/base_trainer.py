from abc import ABC, abstractmethod
import os
import joblib

class BaseTrainer(ABC):
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.classes = []

    @abstractmethod
    def train(self, X, y, classes):
        """
        Train the model.
        X: numpy array of image data [N, H, W, C]
        y: numpy array of labels [N]
        classes: list of class names
        """
        pass

    @abstractmethod
    def predict(self, image_data):
        """
        Predict the class of an image.
        image_data: numpy array of the image [H, W, C]
        Returns: dict of {class_name: probability}
        """
        pass

    def save(self):
        """Save the model and classes to disk."""
        if self.model is not None:
            joblib.dump({'model': self.model, 'classes': self.classes}, self.model_path)

    def load(self):
        """Load the model and classes from disk."""
        if os.path.exists(self.model_path):
            data = joblib.load(self.model_path)
            self.model = data['model']
            self.classes = data['classes']
            return True
        return False

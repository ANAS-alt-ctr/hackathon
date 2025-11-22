from .base_trainer import BaseTrainer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

class SklearnTrainer(BaseTrainer):
    def __init__(self, model_type, model_path):
        super().__init__(model_path)
        self.model_type = model_type
        
        if model_type == 'logistic_regression':
            # Robust default parameters
            self.model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
        elif model_type == 'random_forest':
            # Robust default parameters
            self.model = RandomForestClassifier(n_estimators=100, max_depth=None)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train(self, X, y, classes):
        self.classes = classes
        
        # Flatten images: [N, H, W, C] -> [N, H*W*C]
        n_samples = X.shape[0]
        X_flat = X.reshape(n_samples, -1)
        
        self.model.fit(X_flat, y)
        
        # Calculate metrics
        y_pred = self.model.predict(X_flat)
        acc = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred).tolist()
        
        self.save()
        return {"accuracy": acc, "confusion_matrix": cm}

    def predict(self, image_data):
        if self.model is None:
            self.load()
            
        # Ensure input is a batch: [H, W, C] -> [1, H, W, C]
        if len(image_data.shape) == 3:
            image_data = np.expand_dims(image_data, axis=0)
            
        # Flatten: [1, H*W*C]
        X_flat = image_data.reshape(image_data.shape[0], -1)
        
        probas = self.model.predict_proba(X_flat)[0]
        
        result = {}
        for i, class_name in enumerate(self.classes):
            result[class_name] = float(probas[i])
            
        return result

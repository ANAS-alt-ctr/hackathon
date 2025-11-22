from .base_trainer import BaseTrainer
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import os
import joblib

class CNNTrainer(BaseTrainer):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.model = None
        self.image_size = (128, 128)

    def _build_model(self, num_classes):
        # Data Augmentation
        data_augmentation = models.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])

        # Pre-trained MobileNetV2
        # We use include_top=False to exclude the final classification layer
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(self.image_size[0], self.image_size[1], 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False # Freeze base model weights

        model = models.Sequential([
            layers.Input(shape=(self.image_size[0], self.image_size[1], 3)),
            data_augmentation,
            
            # Preprocessing for MobileNetV2: inputs should be in [-1, 1]
            # Input images are [0, 255].
            # Rescaling(1./127.5, offset=-1) does: (x * 1/127.5) - 1
            # This is a standard Keras layer and serializes safely.
            layers.Rescaling(1./127.5, offset=-1),
            
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self, X, y, classes):
        self.classes = classes
        num_classes = len(classes)
        
        # X is expected to be [0, 255] float32
        X = X.astype('float32')
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = self._build_model(num_classes)
        
        # Early Stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train, 
            epochs=25, 
            validation_data=(X_val, y_val), 
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate
        y_pred_prob = self.model.predict(X_val)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        acc = accuracy_score(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred).tolist()
        
        self.save()
        return {"accuracy": acc, "confusion_matrix": cm, "history": history.history}

    def predict(self, image_data):
        if self.model is None:
            self.load()
            
        # Ensure batch dimension: [H, W, C] -> [1, H, W, C]
        if len(image_data.shape) == 3:
            image_data = np.expand_dims(image_data, axis=0)
            
        # Ensure float32 type
        image_data = image_data.astype('float32')
        
        probas = self.model.predict(image_data)[0]
        
        result = {}
        for i, class_name in enumerate(self.classes):
            result[class_name] = float(probas[i])
            
        return result

    def save(self):
        if self.model is not None:
            # Save Keras model to a separate .keras file
            keras_path = self.model_path.replace('.pkl', '.keras')
            self.model.save(keras_path)
            
            # Save metadata (classes and path to keras model) using joblib
            joblib.dump({
                'classes': self.classes, 
                'keras_path': keras_path
            }, self.model_path)

    def load(self):
        if os.path.exists(self.model_path):
            try:
                data = joblib.load(self.model_path)
                self.classes = data['classes']
                keras_path = data['keras_path']
                
                if os.path.exists(keras_path):
                    # Standard load_model should work now with Rescaling layer
                    self.model = models.load_model(keras_path)
                    return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        return False

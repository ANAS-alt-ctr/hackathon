import streamlit as st
import os
# Set env vars before importing tensorflow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
from PIL import Image
import shutil
from trainers.sklearn_trainer import SklearnTrainer
from trainers.cnn_trainer import CNNTrainer

# --- Configuration ---
DATA_DIR = "data"
MODELS_DIR = "models"
IMAGE_SIZE = (128, 128)

st.set_page_config(page_title="Teachable Machine Clone", layout="wide")

# --- Helper Functions ---
def save_uploaded_image(uploaded_file, class_name):
    """Saves an uploaded file to the class directory."""
    class_dir = os.path.join(DATA_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)
    file_path = os.path.join(class_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def load_dataset():
    """Loads all images and labels from the data directory."""
    images = []
    labels = []
    if not os.path.exists(DATA_DIR):
        return np.array([]), np.array([]), []

    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    
    for idx, class_name in enumerate(classes):
        class_dir = os.path.join(DATA_DIR, class_name)
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            try:
                img = Image.open(file_path).convert('RGB')
                img = img.resize(IMAGE_SIZE)
                images.append(np.array(img))
                labels.append(idx)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
    return np.array(images), np.array(labels), classes

def preprocess_image(image):
    """Prepares a single image for prediction."""
    img = image.convert('RGB')
    img = img.resize(IMAGE_SIZE)
    return np.array(img)

# --- Main UI ---
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose Mode", ["Data Collection", "Training", "Inference"])

if app_mode == "Data Collection":
    st.title("1. Data Collection")
    st.markdown("Create classes and upload images to train your model.")

    # Class Management
    col1, col2 = st.columns([3, 1])
    with col1:
        new_class = st.text_input("Add New Class Name")
    with col2:
        if st.button("Reset All Data", type="primary"):
            st.session_state.confirm_reset = True

    # Reset Confirmation
    if st.session_state.get("confirm_reset", False):
        st.warning("Are you sure? This will delete all images.")
        if st.button("Yes, Delete Everything"):
            if os.path.exists(DATA_DIR):
                shutil.rmtree(DATA_DIR)
            os.makedirs(DATA_DIR, exist_ok=True)
            st.session_state.confirm_reset = False
            st.success("Data reset!")
            st.rerun()
        if st.button("Cancel"):
            st.session_state.confirm_reset = False
            st.rerun()

    # Add Class Logic
    if st.button("Add Class"):
        if new_class:
            os.makedirs(os.path.join(DATA_DIR, new_class), exist_ok=True)
            st.success(f"Class '{new_class}' added!")
            st.rerun()

    # Display Classes
    if os.path.exists(DATA_DIR):
        classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    else:
        classes = []

    if not classes:
        st.info("No classes yet. Add one above!")
    
    # Grid Layout for Classes
    cols = st.columns(len(classes) if classes else 1)
    for i, class_name in enumerate(classes):
        with cols[i % len(cols)]:
            st.subheader(f"Class: {class_name}")
            
            # File Uploader
            uploaded_files = st.file_uploader(f"Upload for {class_name}", accept_multiple_files=True, key=f"uploader_{class_name}")
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    save_uploaded_image(uploaded_file, class_name)
                st.success(f"Saved {len(uploaded_files)} images")
            
            # Camera Input
            camera_img = st.camera_input(f"Cam for {class_name}", key=f"cam_{class_name}")
            if camera_img:
                import time
                timestamp = int(time.time() * 1000)
                camera_img.name = f"cam_{timestamp}.jpg"
                save_uploaded_image(camera_img, class_name)
                st.success("Image captured!")

            # Count
            count = len(os.listdir(os.path.join(DATA_DIR, class_name)))
            st.caption(f"{count} images")

elif app_mode == "Training":
    st.title("2. Training")
    st.markdown("Train a model on your collected data.")

    model_type = st.selectbox("Select Model", ["Logistic Regression", "Random Forest", "CNN (TensorFlow)"])
    
    if st.button("Train Model"):
        with st.spinner("Loading data and training..."):
            X, y, classes = load_dataset()
            
            # Validation
            if len(classes) < 2:
                st.error("Need at least 2 classes to train!")
            elif len(X) == 0:
                st.error("No images found!")
            else:
                # Check distribution
                dist = {c: list(y).count(i) for i, c in enumerate(classes)}
                st.write("Data Distribution:", dist)
                
                if any(count < 5 for count in dist.values()):
                    st.warning("Warning: Some classes have very few images (<5). Results might be poor.")

                # Initialize Trainer
                os.makedirs(MODELS_DIR, exist_ok=True)
                model_path = os.path.join(MODELS_DIR, f"{model_type.replace(' ', '_').lower()}.pkl")
                
                if model_type == "Logistic Regression":
                    trainer = SklearnTrainer("logistic_regression", model_path)
                elif model_type == "Random Forest":
                    trainer = SklearnTrainer("random_forest", model_path)
                elif model_type == "CNN (TensorFlow)":
                    trainer = CNNTrainer(model_path)
                
                # Train
                metrics = trainer.train(X, y, classes)
                
                st.success("Training Complete!")
                st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                
                st.subheader("Confusion Matrix")
                st.write(metrics['confusion_matrix'])
                
                if 'history' in metrics:
                    st.subheader("Training History")
                    st.line_chart(metrics['history']['accuracy'])

                if 'history' in metrics:
                    st.subheader("Training History")
                    st.line_chart(metrics['history']['accuracy'])

elif app_mode == "Inference":
    st.title("3. Inference")
    st.markdown("Test your trained model.")
    
    model_type = st.selectbox("Select Model to Use", ["Logistic Regression", "Random Forest", "CNN (TensorFlow)"])
    model_path = os.path.join(MODELS_DIR, f"{model_type.replace(' ', '_').lower()}.pkl")
    
    if not os.path.exists(model_path):
        st.warning("Model not trained yet! Go to the Training tab.")
    else:
        # Load Trainer
        if model_type == "Logistic Regression":
            trainer = SklearnTrainer("logistic_regression", model_path)
        elif model_type == "Random Forest":
            trainer = SklearnTrainer("random_forest", model_path)
        elif model_type == "CNN (TensorFlow)":
            trainer = CNNTrainer(model_path)
            
        if trainer.load():
            st.success(f"Loaded {model_type}")
            
            # Input Method
            input_method = st.radio("Input Method", ["Upload Image", "Webcam"])
            image_to_predict = None
            
            if input_method == "Upload Image":
                uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
                if uploaded_file:
                    image_to_predict = Image.open(uploaded_file)
                    st.image(image_to_predict, caption="Uploaded Image", width=300)
                    
            elif input_method == "Webcam":
                camera_img = st.camera_input("Take a picture")
                if camera_img:
                    image_to_predict = Image.open(camera_img)
                    
            if image_to_predict:
                img_array = preprocess_image(image_to_predict)
                predictions = trainer.predict(img_array)
                
                st.subheader("Predictions")
                for class_name, prob in predictions.items():
                    st.write(f"**{class_name}**: {prob:.2%}")
                    st.progress(prob)
                
                best_class = max(predictions, key=predictions.get)
                st.success(f"Predicted Class: **{best_class}**")
        else:
            st.error("Failed to load model. Please retrain.")

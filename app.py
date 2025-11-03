import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import cv2


# Page configuration
st.set_page_config(
    page_title="Traffic Sign Classifier",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Class labels dictionary
classes = { 
    0:'Speed limit (20km/h)',
    1:'Speed limit (30km/h)', 
    2:'Speed limit (50km/h)', 
    3:'Speed limit (60km/h)', 
    4:'Speed limit (70km/h)', 
    5:'Speed limit (80km/h)', 
    6:'End of speed limit (80km/h)', 
    7:'Speed limit (100km/h)', 
    8:'Speed limit (120km/h)', 
    9:'No passing', 
    10:'No passing veh over 3.5 tons', 
    11:'Right-of-way at intersection', 
    12:'Priority road', 
    13:'Yield', 
    14:'Stop', 
    15:'No vehicles', 
    16:'Veh > 3.5 tons prohibited', 
    17:'No entry', 
    18:'General caution', 
    19:'Dangerous curve left', 
    20:'Dangerous curve right', 
    21:'Double curve', 
    22:'Bumpy road', 
    23:'Slippery road', 
    24:'Road narrows on the right', 
    25:'Road work', 
    26:'Traffic signals', 
    27:'Pedestrians', 
    28:'Children crossing', 
    29:'Bicycles crossing', 
    30:'Beware of ice/snow',
    31:'Wild animals crossing', 
    32:'End speed + passing limits', 
    33:'Turn right ahead', 
    34:'Turn left ahead', 
    35:'Ahead only', 
    36:'Go straight or right', 
    37:'Go straight or left', 
    38:'Keep right', 
    39:'Keep left', 
    40:'Roundabout mandatory', 
    41:'End of no passing', 
    42:'End no passing veh > 3.5 tons' 
}

# Load model with caching
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = keras.models.load_model('model/best_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Make sure 'best_model.h5' is in the 'model/' folder")
        return None

def preprocess_image(image):
    """
    Preprocess the image to match training preprocessing:
    - Resize to 30x30
    - Convert to RGB
    - Normalize to 0-1 range
    """
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Convert to RGB if needed (in case of RGBA or grayscale)
        if len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        # Resize to 30x30
        img_resized = cv2.resize(img_array, (30, 30))
        
        # Normalize to 0-1
        img_normalized = img_resized / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
    
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def predict(model, image):
    """Make prediction on the preprocessed image"""
    try:
        # Get predictions
        predictions = model.predict(image, verbose=0)
        
        # Get top 3 predictions
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        top_3_probs = predictions[0][top_3_idx]
        
        results = []
        for idx, prob in zip(top_3_idx, top_3_probs):
            results.append({
                'class': classes[idx],
                'confidence': float(prob * 100)
            })
        
        return results
    
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

# Main App
def main():
    model = load_model()
    
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "Home"

    # Sidebar navigation
    with st.sidebar:
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = "Home"
        if st.button("⚙️ About Model", use_container_width=True):
            st.session_state.page = "About Model"
        if st.button("📊 About Data", use_container_width=True):
            st.session_state.page = "About Data"
    
    # Page routing
    if st.session_state.page == "Home":
        show_home_page(model)
    elif st.session_state.page == "About Model":
        show_model_page()
    elif st.session_state.page == "About Data":
        show_data_page()

def show_home_page(model):
    """Display the home page with classifier functionality"""
    st.title("🚦 Traffic Sign Classifier")
    st.markdown("Upload a traffic sign image for automated classification using deep learning")
    st.markdown("**Model Performance:** 97% accuracy on GTSRB test dataset")
    
    if model is None:
        st.error("Model could not be loaded. Please check if 'best_model.h5' exists in the 'model/' folder.")
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Select a traffic sign image",
        type=['jpg', 'jpeg', 'png', 'ppm'],
        help="Upload a clear image of a traffic sign for classification"
    )
    
    if uploaded_file is not None:
        # Create two columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, width="stretch")
        
        with col2:
            st.subheader("Classification Results")
            
            # Add a predict button
            if st.button("Classify Traffic Sign", type="primary", width="stretch"):
                with st.spinner("Processing image..."):
                    # Preprocess image
                    processed_image = preprocess_image(image)
                    
                    if processed_image is not None:
                        # Make prediction
                        results = predict(model, processed_image)
                        
                        if results is not None:
                            # Display results
                            st.success("Classification complete")
                            
                            # Top prediction (highlighted)
                            st.markdown("---")
                            st.markdown("**Primary Classification**")
                            st.metric(
                                label="Predicted Sign",
                                value=results[0]['class'],
                                delta=f"{results[0]['confidence']:.1f}% confidence"
                            )
                            
                            # Top 3 predictions
                            st.markdown("---")
                            st.markdown("**Top 3 Predictions**")
                            
                            for i, result in enumerate(results, 1):
                                confidence = result['confidence']
                                # Create progress bar for confidence
                                st.markdown(f"**{i}. {result['class']}**")
                                st.progress(confidence / 100)
                                st.caption(f"Confidence: {confidence:.1f}%")
                                st.markdown("")
    
    else:
        # Show example message when no file is uploaded
        st.info("Please upload a traffic sign image to begin classification")
        
        # Optional: Show sample images or instructions
        st.markdown("---")
        st.markdown("### Best Practices for Optimal Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📷 Image Quality**")
            st.markdown("Use well-lit, high-resolution images with minimal blur")
        
        with col2:
            st.markdown("**🎯 Sign Position**")
            st.markdown("Ensure the traffic sign is centered and clearly visible")
        
        with col3:
            st.markdown("**Background**")
            st.markdown("Minimize visual clutter around the sign for better accuracy")

def show_model_page():
    """Display detailed information about the model"""
    col1, col2 = st.columns([0.08, 0.1])
    with col1:
        st.title("About the Model")
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103832.png", width=100)
    st.markdown("[Check the Notebook for more details ↗](https://www.kaggle.com/code/aakcodebreaker/german-traffic-signs-classification-97)")
    st.markdown("---")
    
    # Model Architecture Section
    st.header("Model Architecture")
    st.markdown("""
    This traffic sign classifier is built using a custom **Convolutional Neural Network (CNN)** 
    designed specifically for the GTSRB dataset.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Network Structure")
        st.markdown("""
        **Convolutional Blocks:**
        - Block 1: Conv2D (16 filters, 5×5) → Conv2D (32 filters, 5×5)
        - Block 2: Conv2D (64 filters, 3×3) → Conv2D (128 filters, 3×3)
        
        **Regularization:**
        - Batch Normalization after each block
        - MaxPooling2D (2×2) for downsampling
        - Dropout (0.25 after conv blocks, 0.5 after dense)
        
        **Classification Head:**
        - Global Average Pooling
        - Dense layer (512 units, ReLU)
        - Output layer (43 units, Softmax)
        """)
    
    with col2:
        st.subheader("Training Configuration")
        st.markdown("""
        **Optimizer:** Adam
        - Learning rate: 0.001
        - Adaptive learning rate with ReduceLROnPlateau
        
        **Loss Function:** Categorical Crossentropy
        - Label smoothing: 0.08
        
        **Data Augmentation:**
        - Rotation: ±15 degrees
        - Zoom: ±15%
        - Width/Height shift: ±10%
        - Shear: ±15%
        
        **Training Details:**
        - Epochs: 20
        - Batch size: 32
        - Early stopping with patience: 8
        """)
    
    st.markdown("---")
    
    # Performance Metrics
    st.header("Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Training Accuracy", "95.7%")
        st.caption("Accuracy on training set")
    
    with col2:
        st.metric("Validation Accuracy", "99.5%")
        st.caption("Accuracy on validation set")
    
    with col3:
        st.metric("Test Accuracy", "97.0%")
        st.caption("Accuracy on unseen test data")
    
    st.markdown("---")
    
    # Technical Specifications
    st.header("Technical Specifications")
    
    specs_col1, specs_col2 = st.columns(2)
    
    with specs_col1:
        st.markdown("""
        **Framework & Libraries:**
        - TensorFlow/Keras
        - OpenCV for image processing
        - NumPy for numerical operations
        
        **Model Size:**
        - Total parameters: 197,195
        - Trainable parameters: 195,851
        - Model file size: ~770 KB
        """)
    
    with specs_col2:
        st.markdown("""
        **Input Specifications:**
        - Image size: 30×30 pixels
        - Color channels: 3 (RGB)
        - Normalization: [0, 1] range
        
        **Output:**
        - 43 classes (traffic sign types)
        - Softmax probability distribution
        """)

def show_data_page():
    """Display information about the dataset"""
    st.title("About the Dataset")
    st.markdown("[Check the Dataset for more details ↗](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)")
    st.markdown("---")
    
    # Dataset Overview
    st.header("Dataset Overview")
    st.markdown("""
    The **German Traffic Sign Recognition Benchmark (GTSRB)** is a large-scale dataset 
    used for multi-class classification of traffic signs. It contains real-world images 
    captured under varying lighting conditions, weather, and viewing angles.
    """)
    st.markdown("![](http://benchmark.ini.rub.de/Images/gtsrb/0.png)![](http://benchmark.ini.rub.de/Images/gtsrb/1.png)![](http://benchmark.ini.rub.de/Images/gtsrb/2.png)![](http://benchmark.ini.rub.de/Images/gtsrb/3.png)![](http://benchmark.ini.rub.de/Images/gtsrb/4.png)![](http://benchmark.ini.rub.de/Images/gtsrb/5.png)![](http://benchmark.ini.rub.de/Images/gtsrb/12.png)![](http://benchmark.ini.rub.de/Images/gtsrb/11.png)![](http://benchmark.ini.rub.de/Images/gtsrb/8.png)")
    st.markdown("---")
    
    # Dataset Statistics
    st.header("Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Images", "51,839")
    
    with col2:
        st.metric("Number of Classes", "43")
    
    with col3:
        st.metric("Training Images", "27,446")
    
    with col4:
        st.metric("Test Images", "12,630")
    
    st.markdown("---")
    
    # Data Split Information
    st.header("Data Split")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training & Validation")
        st.markdown("""
        **Training Set:** 27,446 images (70%)
        - Used for model learning
        - Applied data augmentation
        
        **Validation Set:** 11,763 images (30%)
        - Used for hyperparameter tuning
        - No augmentation applied
        
        **Split Method:** Stratified random split
        """)
    
    with col2:
        st.subheader("Test Set")
        st.markdown("""
        **Test Set:** 12,630 images
        - Completely unseen during training
        - Used for final model evaluation
        - Represents real-world performance
        
        **Final Test Accuracy:** 97.01%
        """)
    
    st.markdown("---")
    
    # Image Specifications
    st.header("Image Specifications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Processing Pipeline")
        st.markdown("""
        **Original Images:**
        - Variable sizes
        - Real-world capture conditions
        - Different lighting and weather
        
        **Preprocessing Steps:**
        1. Resize to 30×30 pixels
        2. Convert to RGB color space
        3. Normalize to [0, 1] range
        """)
    
    with col2:
        st.subheader("Data Characteristics")
        st.markdown("""
        **Challenges:**
        - Varying illumination
        - Partial occlusions
        - Different viewing angles
        - Weather conditions (fog, rain)
        - Motion blur
        
        **Real-world Application:**
        - Autonomous vehicles
        - Driver assistance systems
        - Traffic monitoring
        """)
    
    st.markdown("---")
    
    # Traffic Sign Categories
    st.header("Traffic Sign Categories")
    st.markdown("""
    The dataset includes 43 different types of traffic signs organized into several categories:
    """)
    
    cat_col1, cat_col2, cat_col3 = st.columns(3)
    
    with cat_col1:
        st.markdown("""
        **Speed Limits (9 classes)**
        - 20, 30, 50, 60, 70, 80, 100, 120 km/h
        - End of speed limit (80 km/h)
        
        **Prohibitory Signs (8 classes)**
        - No passing
        - No vehicles
        - No entry
        - Weight limits
        """)
    
    with cat_col2:
        st.markdown("""
        **Danger Warnings (11 classes)**
        - General caution
        - Dangerous curves
        - Slippery road
        - Road work
        - Pedestrians/Children crossing
        - Wild animals
        
        **Mandatory Signs (5 classes)**
        - Keep right/left
        - Go straight or right/left
        - Roundabout mandatory
        """)
    
    with cat_col3:
        st.markdown("""
        **Priority Signs (5 classes)**
        - Priority road
        - Yield
        - Stop
        - Right-of-way at intersection
        
        **Directional Signs (5 classes)**
        - Turn right/left ahead
        - Ahead only
        - End of no passing zones
        """)

if __name__ == "__main__":
    main()
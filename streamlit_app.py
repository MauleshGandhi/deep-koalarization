import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os
import matplotlib.pyplot as plt
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Deep Koalarization: Manga Colorization",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .result-container {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f9f9f9;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_colorization_model(model_path):
    """Load the trained colorization model"""
    try:
        if not os.path.exists(model_path):
            st.warning(f"Model file not found: {model_path}")
            return None
        
        # Load model without compilation to avoid optimizer issues
        model = tf.keras.models.load_model(model_path, compile=False)
        st.success(f"✅ Model loaded successfully: {Path(model_path).name}")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def preprocess_image_for_model(image_pil):
    """
    Preprocess PIL image for the colorization model
    Following the exact preprocessing from your original code
    """
    try:
        # Convert PIL to numpy array
        image_np = np.array(image_pil)
        
        # If grayscale, convert to 3-channel by replicating
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        elif len(image_np.shape) == 3 and image_np.shape[2] == 4:
            # Handle RGBA
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
        elif image_np.shape[2] == 3:
            # Convert RGB to BGR for OpenCV
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Step 1: Resize to 299x299 for Inception preprocessing
        img_299 = cv2.resize(image_np, (299, 299))
        
        # Step 2: Apply Inception preprocessing
        img_299 = tf.keras.applications.inception_resnet_v2.preprocess_input(img_299.astype(np.float32))
        
        # Step 3: Extract L channel and create pretrained input
        L_channel = img_299[:, :, 0]  # L channel after preprocessing
        X_pretrained = np.repeat(L_channel[:, :, np.newaxis], 3, axis=2)
        X_pretrained = np.expand_dims(X_pretrained, axis=0)  # Add batch dimension
        
        # Step 4: Resize to 224x224 for encoder
        img_224 = cv2.resize(img_299, (224, 224))
        X_encoder = img_224[:, :, 0]  # L channel
        X_encoder = np.expand_dims(X_encoder, axis=0)  # Add batch dimension
        X_encoder = np.expand_dims(X_encoder, axis=3)  # Add channel dimension
        
        return X_encoder, X_pretrained, img_299
        
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return None, None, None

def colorize_image(model, X_encoder, X_pretrained, original_shape):
    """
    Colorize image using the trained model
    Following the exact post-processing from your original code
    """
    try:
        # Predict ab channels
        Y_pred = model.predict([X_encoder, X_pretrained], verbose=0)
        
        # Get predictions and encoder input
        ab_pred = Y_pred[0]  # Remove batch dimension
        L_channel = X_encoder[0, :, :, 0]  # Remove batch and channel dimensions
        
        # Create LAB image following your original post-processing
        new_img = np.zeros((224, 224, 3))
        
        # Normalize L channel from [-1, 1] to [0, 1] then scale to [0, 255]
        L_normalized = (L_channel + 1) / 2
        ab_normalized = (ab_pred + 1) / 2
        
        new_img[:, :, 0] = L_normalized * 255
        new_img[:, :, 1:] = ab_normalized * 255
        
        # Ensure values are in correct range
        new_img = np.clip(new_img, 0, 255).astype(np.uint8)
        
        # Convert LAB to RGB
        colorized_rgb = cv2.cvtColor(new_img, cv2.COLOR_LAB2RGB)
        
        return colorized_rgb
        
    except Exception as e:
        st.error(f"Error during colorization: {str(e)}")
        return None

def create_comparison_plot(original_image, colorized_image):
    """Create a side-by-side comparison plot"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    if len(original_image.shape) == 3:
        # Convert BGR to RGB for display
        original_display = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    else:
        original_display = original_image
        
    axes[0].imshow(original_display)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis('off')
    
    # Colorized image
    axes[1].imshow(colorized_image)
    axes[1].set_title("Colorized Image", fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    return fig

def main():
    # Main header
    st.markdown('<h1 class="main-header">🎨 Deep Koalarization: Manga Colorization</h1>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    ## Live Demo - Production Model
    
    This is a **working demonstration** of the Deep Koalarization approach for automatic manga colorization. 
    The app uses your trained .h5 models to actually colorize uploaded images in real-time.
    """)
    
    # Model selection sidebar
    st.sidebar.markdown("### 🎯 Model Selection")
    
    # Define available models with their paths
    models_dir = Path("models")  # Adjust path as needed
    available_models = {
        "Koala_00001_20.h5": {
            "name": "Koala 0.00001 LR (20 epochs)",
            "lr": "0.00001",
            "epochs": 20,
            "accuracy": "0.659",
            "loss": "0.0116",
            "description": "Best performing model - Conservative learning rate"
        },
        "Koala_0001_50.h5": {
            "name": "Koala 0.0001 LR (50 epochs)", 
            "lr": "0.0001",
            "epochs": 50,
            "accuracy": "0.649",
            "loss": "0.0128",
            "description": "Balanced training - Good stability"
        },
        "Koala_0005_100.h5": {
            "name": "Koala 0.0005 LR (100 epochs)",
            "lr": "0.0005", 
            "epochs": 100,
            "accuracy": "0.649",
            "loss": "0.0138",
            "description": "Extended training - Detailed features"
        },
        "Koala_0005_40.h5": {
            "name": "Koala 0.0005 LR (40 epochs)",
            "lr": "0.0005",
            "epochs": 40, 
            "accuracy": "0.649",
            "loss": "0.0138",
            "description": "Moderate training - Good balance"
        },
        "Koala_01_20.h5": {
            "name": "Koala 0.01 LR (20 epochs)",
            "lr": "0.01",
            "epochs": 20,
            "accuracy": "0.633", 
            "loss": "0.965",
            "description": "Fast convergence - Quick test model"
        }
    }
    
    # Check which models are actually available
    available_model_files = []
    for model_file in available_models.keys():
        model_path = models_dir / model_file
        if model_path.exists():
            available_model_files.append(model_file)
        else:
            # Also check in root directory
            if Path(model_file).exists():
                available_model_files.append(model_file)
    
    if not available_model_files:
        st.sidebar.error("❌ No model files found. Please ensure .h5 files are in the models/ directory.")
        st.error("""
        **Models not found!** 
        
        Please ensure your trained .h5 model files are placed in a `models/` directory:
        - Koala_00001_20.h5
        - Koala_0001_50.h5  
        - Koala_0005_100.h5
        - Koala_0005_40.h5
        - Koala_01_20.h5
        """)
        return
    
    selected_model_file = st.sidebar.selectbox(
        "Choose a trained model:",
        available_model_files,
        help="Select from your available trained models"
    )
    
    model_info = available_models[selected_model_file]
    
    # Display model details
    st.sidebar.markdown("#### 📊 Model Details")
    st.sidebar.markdown(f"""
    **{model_info['name']}**
    - Learning Rate: {model_info['lr']}
    - Training Epochs: {model_info['epochs']}
    - Test Accuracy: {model_info['accuracy']}
    - Test Loss: {model_info['loss']}
    
    *{model_info['description']}*
    """)
    
    # Load selected model
    model_path = models_dir / selected_model_file
    if not model_path.exists():
        model_path = Path(selected_model_file)  # Try root directory
        
    model = load_colorization_model(str(model_path))
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="section-header">📁 Upload Image</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image to colorize",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Upload grayscale or color images. Works best with manga-style artwork."
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)
            
            # Show image info
            st.markdown(f"""
            **Image Info:**
            - Size: {image.size[0]} × {image.size[1]} pixels
            - Mode: {image.mode}
            - Format: {image.format}
            """)
    
    with col2:
        st.markdown('<div class="section-header">🎨 Colorization Result</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None and model is not None:
            if st.button("🚀 Colorize Image", type="primary", use_container_width=True):
                
                with st.spinner("Processing image... This may take a few seconds."):
                    # Preprocess image
                    X_encoder, X_pretrained, original_processed = preprocess_image_for_model(image)
                    
                    if X_encoder is not None and X_pretrained is not None:
                        # Colorize
                        colorized = colorize_image(model, X_encoder, X_pretrained, image.size)
                        
                        if colorized is not None:
                            # Display colorized result
                            st.image(colorized, caption="Colorized Result", use_column_width=True)
                            
                            # Success message
                            st.markdown('<div class="success-box">✅ Colorization completed successfully!</div>', unsafe_allow_html=True)
                            
                            # Provide download option
                            colorized_pil = Image.fromarray(colorized)
                            buf = io.BytesIO()
                            colorized_pil.save(buf, format='PNG')
                            byte_im = buf.getvalue()
                            
                            st.download_button(
                                label="💾 Download Colorized Image",
                                data=byte_im,
                                file_name=f"colorized_{uploaded_file.name.split('.')[0]}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                            
                            # Show processing details
                            with st.expander("🔍 Processing Details"):
                                st.markdown(f"""
                                **Model Used:** {model_info['name']}
                                
                                **Processing Steps:**
                                1. Image resized to 299×299 for Inception-ResNet-v2
                                2. Applied Inception preprocessing to LAB color space
                                3. L channel extracted for both encoder (224×224) and pretrained (299×299) inputs
                                4. Model predicted a and b channels
                                5. Combined L + predicted ab channels 
                                6. Converted LAB back to RGB for display
                                
                                **Input Shapes:**
                                - Encoder input: (1, 224, 224, 1)
                                - Pretrained input: (1, 299, 299, 3)
                                - Output: (1, 224, 224, 2)
                                """)
        elif uploaded_file is not None and model is None:
            st.error("❌ Model not loaded. Please check if the model file exists.")
        else:
            st.info("👆 Upload an image and click 'Colorize Image' to see the AI colorization in action!")
    
    # Technical details section
    st.markdown("---")
    st.markdown('<div class="section-header">🔬 How It Works</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🏗️ Dual Architecture**
        - **CNN Encoder**: Extracts spatial features from L channel (224×224)
        - **Inception-ResNet-v2**: Provides semantic understanding (299×299) 
        - **Fusion Layer**: Combines both feature types
        - **CNN Decoder**: Upsamples to predict ab channels
        """)
    
    with col2:
        st.markdown("""
        **🎨 LAB Color Space**
        - **L Channel**: Luminance (brightness) - model input
        - **a Channel**: Green-Red axis - predicted by model
        - **b Channel**: Blue-Yellow axis - predicted by model
        - **Advantage**: Separates color from brightness
        """)
    
    with col3:
        st.markdown("""
        **⚙️ Training Details**
        - **Dataset**: ImageNet + Danbooru2020small
        - **Loss Function**: Mean Squared Error
        - **Optimizer**: Adam with various learning rates
        - **Best Model**: 0.00001 LR, 20 epochs
        """)
    
    # Model comparison table
    if len(available_model_files) > 1:
        st.markdown('<div class="section-header">📊 Model Comparison</div>', unsafe_allow_html=True)
        
        comparison_data = []
        for model_file in available_model_files:
            info = available_models[model_file]
            comparison_data.append({
                "Model": info['name'],
                "Learning Rate": info['lr'],
                "Epochs": info['epochs'],
                "Test Accuracy": info['accuracy'],
                "Test Loss": info['loss']
            })
        
        st.table(comparison_data)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p><strong>Deep Koalarization: Manga Colorization</strong></p>
        <p>🎓 <em>Anish Mathur, Maulesh Gandhi, Pavan Kondooru | IIIT Hyderabad</em></p>
        <p>📄 Based on "Deep Koalarization: Image Colorization using CNNs and Inception-Resnet-v2"</p>
        <p>🔗 <a href="https://github.com/your-username/deep-koalarization">GitHub Repository</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

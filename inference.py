"""
Deep Koalarization: Inference Script
Authors: Anish Mathur, Maulesh Gandhi, Pavan Kondooru

This script loads a trained model and performs colorization on test images.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import tensorflow as tf
from PIL import Image

def load_colorization_model(model_path):
    """
    Load a trained colorization model

    Args:
        model_path: Path to the saved .h5 model file

    Returns:
        Loaded Keras model
    """
    try:
        model = load_model(model_path, compile=False)
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def preprocess_single_image(image_path, target_size_encoder=(224, 224), 
                           target_size_pretrained=(299, 299)):
    """
    Preprocess a single image for colorization

    Args:
        image_path: Path to the input image
        target_size_encoder: Target size for encoder input
        target_size_pretrained: Target size for pretrained model input

    Returns:
        Tuple of preprocessed inputs (X_encoder, X_pretrained)
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Convert to LAB color space
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Preprocess for Inception-ResNet-v2
    img_299 = cv2.resize(img_lab, target_size_pretrained)
    img_299 = tf.keras.applications.inception_resnet_v2.preprocess_input(img_299)

    # Extract L channel and create pretrained input
    L = img_299[:, :, 0]
    X_pretrained = np.repeat(L[:, :, np.newaxis], 3, axis=2)
    X_pretrained = np.expand_dims(X_pretrained, axis=0)

    # Create encoder input
    img_224 = cv2.resize(img_lab, target_size_encoder)
    X_encoder = img_224[:, :, 0]  # L channel only
    X_encoder = np.expand_dims(X_encoder, axis=0)

    return X_encoder, X_pretrained, img_lab

def colorize_image(model, X_encoder, X_pretrained, original_lab_shape):
    """
    Colorize an image using the trained model

    Args:
        model: Trained colorization model
        X_encoder: Preprocessed encoder input
        X_pretrained: Preprocessed pretrained model input
        original_lab_shape: Shape of original LAB image

    Returns:
        Colorized RGB image
    """
    # Predict ab channels
    ab_predicted = model.predict([X_encoder, X_pretrained])

    # Get the predicted ab channels
    ab_pred = ab_predicted[0]  # Remove batch dimension

    # Resize L channel to match ab prediction
    L_resized = cv2.resize(X_encoder[0], (ab_pred.shape[1], ab_pred.shape[0]))

    # Combine L and predicted ab channels
    lab_output = np.zeros((ab_pred.shape[0], ab_pred.shape[1], 3))
    lab_output[:, :, 0] = L_resized
    lab_output[:, :, 1:] = ab_pred

    # Convert LAB to RGB
    lab_output = lab_output.astype(np.uint8)
    rgb_output = cv2.cvtColor(lab_output, cv2.COLOR_LAB2RGB)

    # Resize to original image size
    if original_lab_shape:
        rgb_output = cv2.resize(rgb_output, (original_lab_shape[1], original_lab_shape[0]))

    return rgb_output

def process_single_image(model_path, image_path, output_path=None, show_result=True):
    """
    Process a single image for colorization

    Args:
        model_path: Path to trained model
        image_path: Path to input image
        output_path: Path to save colorized image (optional)
        show_result: Whether to display the result

    Returns:
        Colorized RGB image
    """
    # Load model
    model = load_colorization_model(model_path)
    if model is None:
        return None

    # Preprocess image
    X_encoder, X_pretrained, original_lab = preprocess_single_image(image_path)

    # Colorize
    colorized = colorize_image(model, X_encoder, X_pretrained, original_lab.shape)

    # Save result if output path provided
    if output_path:
        colorized_bgr = cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, colorized_bgr)
        print(f"Colorized image saved to: {output_path}")

    # Display result
    if show_result:
        plt.figure(figsize=(12, 6))

        # Original image
        original_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        plt.subplot(1, 2, 1)
        plt.imshow(original_rgb)
        plt.title("Original Image")
        plt.axis('off')

        # Colorized image
        plt.subplot(1, 2, 2)
        plt.imshow(colorized)
        plt.title("Colorized Image")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    return colorized

def batch_colorize(model_path, input_folder, output_folder):
    """
    Colorize all images in a folder

    Args:
        model_path: Path to trained model
        input_folder: Folder containing input images
        output_folder: Folder to save colorized images
    """
    # Load model
    model = load_colorization_model(model_path)
    if model is None:
        return

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(input_folder) 
                  if any(f.lower().endswith(ext) for ext in image_extensions)]

    print(f"Found {len(image_files)} images to process")

    for i, image_file in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {image_file}")

        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, f"colorized_{image_file}")

        try:
            # Process image
            X_encoder, X_pretrained, original_lab = preprocess_single_image(input_path)
            colorized = colorize_image(model, X_encoder, X_pretrained, original_lab.shape)

            # Save result
            colorized_bgr = cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, colorized_bgr)

        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")

    print("Batch colorization completed!")

def evaluate_model(model_path, test_folder, num_samples=5):
    """
    Evaluate model on a few test samples

    Args:
        model_path: Path to trained model
        test_folder: Folder containing test images
        num_samples: Number of samples to evaluate
    """
    # Load model
    model = load_colorization_model(model_path)
    if model is None:
        return

    # Get test images
    image_files = [f for f in os.listdir(test_folder) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Sample random images
    import random
    sample_files = random.sample(image_files, min(num_samples, len(image_files)))

    plt.figure(figsize=(15, 3 * len(sample_files)))

    for i, image_file in enumerate(sample_files):
        image_path = os.path.join(test_folder, image_file)

        # Process image
        X_encoder, X_pretrained, original_lab = preprocess_single_image(image_path)
        colorized = colorize_image(model, X_encoder, X_pretrained, original_lab.shape)

        # Display original and colorized
        original_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        plt.subplot(len(sample_files), 2, 2*i + 1)
        plt.imshow(original_rgb)
        plt.title(f"Original: {image_file}")
        plt.axis('off')

        plt.subplot(len(sample_files), 2, 2*i + 2)
        plt.imshow(colorized)
        plt.title(f"Colorized: {image_file}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "Koala_0005_20.h5"  # Update with your model path

    # Example usage:

    # 1. Process a single image
    # process_single_image(MODEL_PATH, "test_image.jpg", "colorized_output.jpg")

    # 2. Batch process images
    # batch_colorize(MODEL_PATH, "input_folder/", "output_folder/")

    # 3. Evaluate model on test samples
    # evaluate_model(MODEL_PATH, "test_folder/", num_samples=3)

    print("Inference script ready!")
    print("Update MODEL_PATH and uncomment the desired function calls.")

"""
Deep Koalarization: Manga Colorization Training Script
Authors: Anish Mathur, Maulesh Gandhi, Pavan Kondooru

This script trains the Deep Koalarization model for manga colorization
using a combination of Inception-ResNet-v2 and a custom encoder-decoder architecture.
"""

import os
import cv2
import numpy as np
from keras.layers import Input, Concatenate, Conv2D, UpSampling2D
from keras.models import Model
from keras.applications.inception_resnet_v2 import InceptionResNetV2
import keras.backend as K
import tensorflow as tf
from keras.models import load_model
from tensorflow import keras

def create_model(input_shape_encoder=(224, 224, 1), 
                input_shape_pretrained=(299, 299, 3),
                learning_rate=0.0005):
    """
    Create the Deep Koalarization model architecture

    Args:
        input_shape_encoder: Shape for encoder input (L channel)
        input_shape_pretrained: Shape for pre-trained model input
        learning_rate: Learning rate for optimizer

    Returns:
        Compiled Keras model
    """

    # Define inputs
    input_encoder = Input(shape=input_shape_encoder, name='input_encoder')
    input_pretrained = Input(shape=input_shape_pretrained, name='input_pretrained')

    # Encoder branch
    encoder_conv1 = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation='relu', 
                          kernel_initializer='he_uniform', padding='same')(input_encoder)
    encoder_conv2 = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', 
                          kernel_initializer='he_uniform', padding='same')(encoder_conv1)
    encoder_conv3 = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), activation='relu', 
                          kernel_initializer='he_uniform', padding='same')(encoder_conv2)
    encoder_conv4 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', 
                          kernel_initializer='he_uniform', padding='same')(encoder_conv3)
    encoder_conv5 = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), activation='relu', 
                          kernel_initializer='he_uniform', padding='same')(encoder_conv4)
    encoder_conv6 = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', 
                          kernel_initializer='he_uniform', padding='same')(encoder_conv5)
    encoder_conv7 = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', 
                          kernel_initializer='he_uniform', padding='same')(encoder_conv6)
    encoder_conv8 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', 
                          kernel_initializer='he_uniform', padding='same')(encoder_conv7)

    output1 = encoder_conv8

    # Pre-trained Inception-ResNet-v2 branch
    inception_model = InceptionResNetV2(include_top=True, weights='imagenet', 
                                       input_shape=input_shape_pretrained)
    for layer in inception_model.layers:
        layer.trainable = False

    feat_model = Model(inputs=inception_model.inputs, 
                      outputs=inception_model.layers[-2].output)
    output2 = feat_model(input_pretrained)

    # Fusion layer - combine encoder output with Inception features
    h = output1.shape[1]
    w = output1.shape[2]
    x = K.expand_dims(output2, axis=1)
    x = K.expand_dims(x, axis=2)
    x = K.concatenate([x] * h, axis=1)
    x = K.concatenate([x] * w, axis=2)
    fused_output = Concatenate(axis=3)([output1, x])

    # Fusion convolution
    fusion_conv1 = Conv2D(256, kernel_size=(1, 1), strides=(1, 1), activation='relu', 
                         kernel_initializer='he_uniform', padding='same')(fused_output)
    output3 = fusion_conv1

    # Decoder branch
    decoder_conv1 = Conv2D(128, (3, 3), strides=(1, 1), activation="relu", 
                          kernel_initializer='he_uniform', padding="same")(output3)
    decoder_up1 = UpSampling2D((2, 2))(decoder_conv1)
    decoder_conv2 = Conv2D(64, (3, 3), strides=(1, 1), activation="relu", 
                          kernel_initializer='he_uniform', padding="same")(decoder_up1)
    decoder_conv3 = Conv2D(64, (3, 3), strides=(1, 1), activation="relu", 
                          kernel_initializer='he_uniform', padding="same")(decoder_conv2)
    decoder_up2 = UpSampling2D((2, 2))(decoder_conv3)
    decoder_conv4 = Conv2D(32, (3, 3), strides=(1, 1), activation="relu", 
                          kernel_initializer='he_uniform', padding="same")(decoder_up2)
    decoder_conv5 = Conv2D(2, (3, 3), strides=(1, 1), activation="tanh", 
                          kernel_initializer='he_uniform', padding="same")(decoder_conv4)
    decoder_up3 = UpSampling2D((2, 2))(decoder_conv5)

    # Create and compile model
    model = Model(inputs=[input_encoder, input_pretrained], outputs=decoder_up3)
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])

    return model

def image_generator(folder_path, batch_size=32):
    """
    Generator function for loading and preprocessing images

    Args:
        folder_path: Path to image directory
        batch_size: Batch size for training

    Yields:
        Tuple of ([X_encoder, X_pretrained], y) for training
    """
    image_list = os.listdir(folder_path)
    image_list.sort()

    while True:
        for i in range(0, len(image_list), batch_size):
            batch_image_list = image_list[i:i+batch_size]

            X_pretrained = np.zeros((len(batch_image_list), 299, 299, 3))
            X = np.zeros((len(batch_image_list), 224, 224))
            y = np.zeros((len(batch_image_list), 224, 224, 2))

            for j, image_name in enumerate(batch_image_list):
                img = cv2.imread(folder_path + image_name)
                # Preprocess the image
                img = tf.keras.applications.inception_resnet_v2.preprocess_input(img)
                # Extract luminance component
                L = img[:, :, 0]
                # Stack the luminance component three times for pretrained input
                X_pretrained[j] = np.repeat(L[:, :, np.newaxis], 3, axis=2)
                img = cv2.resize(img, (224, 224))
                X[j] = img[:, :, 0]  # L channel
                y[j] = img[:, :, 1:]  # ab channels

            yield [X, X_pretrained], y

def train_model(train_folder, val_folder, test_folder, 
               learning_rate=0.0005, epochs=20, batch_size=32,
               model_name="koala_model"):
    """
    Train the Deep Koalarization model

    Args:
        train_folder: Path to training images
        val_folder: Path to validation images  
        test_folder: Path to test images
        learning_rate: Learning rate for training
        epochs: Number of training epochs
        batch_size: Training batch size
        model_name: Name for saved model
    """

    print("Creating model...")
    model = create_model(learning_rate=learning_rate)
    print("Model created successfully!")

    # Create data generators
    print("Setting up data generators...")
    train_gen = image_generator(train_folder, batch_size)
    val_gen = image_generator(val_folder, batch_size)
    test_gen = image_generator(test_folder, batch_size)

    # Calculate steps
    train_steps = len(os.listdir(train_folder)) // batch_size
    val_steps = len(os.listdir(val_folder)) // batch_size
    test_steps = len(os.listdir(test_folder)) // batch_size

    print(f"Training steps per epoch: {train_steps}")
    print(f"Validation steps per epoch: {val_steps}")

    # Train the model
    print("Starting training...")
    history = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=val_steps,
        verbose=1
    )

    print("Training completed!")

    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = model.evaluate(test_gen, steps=test_steps)
    print(f"Test loss: {test_results[0]:.6f}")
    print(f"Test accuracy: {test_results[1]:.6f}")

    # Save the model
    model_filename = f"{model_name}_{str(learning_rate).replace('.', '')}_{epochs}.h5"
    model.save(model_filename)
    print(f"Model saved as: {model_filename}")

    return model, history

if __name__ == "__main__":
    # Configuration
    TRAIN_FOLDER = './train_new2/'
    VAL_FOLDER = './val_new2/'
    TEST_FOLDER = './test_new2/'

    LEARNING_RATE = 0.0005
    EPOCHS = 20
    BATCH_SIZE = 32
    MODEL_NAME = "Koala"

    # Train the model
    model, history = train_model(
        TRAIN_FOLDER, VAL_FOLDER, TEST_FOLDER,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        model_name=MODEL_NAME
    )

    print("Training pipeline completed successfully!")

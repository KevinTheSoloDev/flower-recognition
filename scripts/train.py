import os
import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_model(num_classes, image_size=224):
    """
    MobileNetV2 based model
    Perfect for flower icon recognition
    """
    # Load pretrained MobileNetV2
    base_model = keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet"
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom layers on top
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def train_model(config):
    """
    Train the flower recognition model
    """
    image_size = config["model"]["image_size"]
    batch_size = config["model"]["batch_size"]
    epochs = config["model"]["epochs"]
    num_classes = config["project"]["num_classes"]
    dataset_path = config["dataset"]["path"]
    model_output = config["paths"]["models"]
    
    os.makedirs(model_output, exist_ok=True)
    os.makedirs(config["paths"]["logs"], exist_ok=True)
    
    print("\n" + "=" * 50)
    print("🚀 FLOWER RECOGNITION TRAINING")
    print("=" * 50)
    print(f"  Classes   : {num_classes}")
    print(f"  Image Size: {image_size}x{image_size}")
    print(f"  Epochs    : {epochs}")
    print(f"  Batch Size: {batch_size}")
    print("=" * 50 + "\n")
    
    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2]
    )
    
    test_datagen = ImageDataGenerator(
        rescale=1./255
    )
    
    # Load datasets
    train_generator = train_datagen.flow_from_directory(
        os.path.join(dataset_path, "train"),
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical"
    )
    
    test_generator = test_datagen.flow_from_directory(
        os.path.join(dataset_path, "test"),
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical"
    )
    
    # Save class names
    class_names = list(train_generator.class_indices.keys())
    import json
    with open(
        os.path.join(model_output, "classes.json"), "w"
    ) as f:
        json.dump(class_names, f, indent=4)
    print(f"✅ Classes saved: {class_names}\n")
    
    # Build model
    print("🔧 Building model...")
    model = build_model(num_classes, image_size)
    model.summary()
    
    # Callbacks
    callbacks = [
        # Save best model automatically
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_output, "model.h5"),
            save_best_only=True,
            monitor="val_accuracy",
            verbose=1
        ),
        # Stop early if not improving
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when stuck
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=5,
            verbose=1
        )
    ]
    
    # Train!
    print("\n🚀 Training started...\n")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Final accuracy
    final_acc = max(history.history["val_accuracy"])
    print(f"\n✅ Training Complete!")
    print(f"🎯 Best Accuracy: {final_acc * 100:.2f}%")
    print(f"📁 Model saved to: {model_output}")

if __name__ == "__main__":
    config = load_config()
    train_model(config)

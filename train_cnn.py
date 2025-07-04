import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import json
import os
from datetime import datetime
import base64
import io
from PIL import Image

# Configure matplotlib for headless operation
plt.switch_backend('Agg')

# CIFAR-10 class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def array_to_base64_image(image_array):
    """Convert numpy array to base64 encoded PNG image"""
    # Ensure image is in the right format (0-255, uint8)
    if image_array.max() <= 1.0:
        image_array = (image_array * 255).astype(np.uint8)
    else:
        image_array = image_array.astype(np.uint8)
    
    # Convert to PIL Image
    image = Image.fromarray(image_array)
    
    # Convert to base64
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    
    base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{base64_str}"

def load_and_preprocess_data():
    """Load and preprocess CIFAR-10 dataset"""
    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    # Split training data into train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    
    # Store original test images for visualization (sample subset for web app)
    test_sample_size = 200  # Reasonable subset for web display
    indices = np.random.choice(len(x_test), test_sample_size, replace=False)
    x_test_viz = x_test[indices].copy()
    y_test_viz = y_test[indices].copy()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_test_viz = x_test_viz.astype('float32') / 255.0
    
    # Convert labels to categorical one-hot encoding
    y_train = keras.utils.to_categorical(y_train, 10)
    y_val = keras.utils.to_categorical(y_val, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    y_test_viz = y_test_viz.flatten()  # Keep original labels for visualization
    
    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Visualization samples: {len(x_test_viz)}")
    
    return {
        'x_train': x_train,
        'y_train': y_train,
        'x_val': x_val,
        'y_val': y_val,
        'x_test': x_test,
        'y_test': y_test,
        'x_test_viz': x_test_viz,  # Original for visualization
        'y_test_viz': y_test_viz,  # Original labels for visualization
        'test_indices': indices
    }

def create_cnn_model():
    """Create a CNN model for CIFAR-10 classification"""
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    return model

def train_model():
    """Train the CNN model on CIFAR-10 dataset"""
    # Load data
    data = load_and_preprocess_data()
    
    # Create model
    model = create_cnn_model()
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print("Model Architecture:")
    model.summary()
    
    # Data augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )
    datagen.fit(data['x_train'])
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3),
        keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
    
    # Train model
    print("Starting training...")
    history = model.fit(
        datagen.flow(data['x_train'], data['y_train'], batch_size=32),
        epochs=30,  # Reduced for faster training
        validation_data=(data['x_val'], data['y_val']),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(data['x_test'], data['y_test'], verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Generate predictions for classification report
    y_pred = model.predict(data['x_test'])
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(data['y_test'], axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=CLASS_NAMES))
    
    return model, history, test_accuracy, test_loss, data

def save_training_plots(history, output_dir='static'):
    """Save training history plots as static images"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('CIFAR-10 CNN Training Results', fontsize=16)
    
    epochs = range(1, len(history.history['loss']) + 1)
    
    # Plot 1: Training and Validation Loss
    ax1.plot(epochs, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training and Validation Accuracy
    ax2.plot(epochs, history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Loss comparison (final values)
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    ax3.bar(['Training Loss', 'Validation Loss'], [final_train_loss, final_val_loss], 
            color=['blue', 'red'], alpha=0.7)
    ax3.set_title('Final Loss Comparison', fontsize=14)
    ax3.set_ylabel('Loss')
    for i, v in enumerate([final_train_loss, final_val_loss]):
        ax3.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
    
    # Plot 4: Accuracy comparison (final values)
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    ax4.bar(['Training Accuracy', 'Validation Accuracy'], [final_train_acc, final_val_acc], 
            color=['blue', 'red'], alpha=0.7)
    ax4.set_title('Final Accuracy Comparison', fontsize=14)
    ax4.set_ylabel('Accuracy')
    ax4.set_ylim(0, 1)
    for i, v in enumerate([final_train_acc, final_val_acc]):
        ax4.text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save the combined plot
    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training plots saved to: {plot_path}")
    return plot_path

def save_model_info(model, history, test_accuracy, test_loss, output_dir='static'):
    """Save model information and training results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model summary as string
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    
    # Count parameters
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    
    # Prepare model info
    model_info = {
        'model_summary': '\n'.join(model_summary),
        'total_parameters': int(total_params),
        'trainable_parameters': int(trainable_params),
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'training_info': {
            'final_train_loss': float(history.history['loss'][-1]),
            'final_train_accuracy': float(history.history['accuracy'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
            'epochs_trained': len(history.history['loss']),
            'training_date': datetime.now().isoformat()
        },
        'layers': [
            {
                'name': layer.name,
                'type': layer.__class__.__name__,
                'output_shape': str(layer.output.shape) if hasattr(layer, 'output') and layer.output is not None else "Not built"
            }
            for layer in model.layers
        ]
    }
    
    # Save model info as JSON
    info_path = os.path.join(output_dir, 'model_info.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"Model info saved to: {info_path}")
    return info_path

def main():
    """Main training function"""
    print("CIFAR-10 CNN Local Training")
    print("=" * 50)
    
    # Train model
    model, history, test_accuracy, test_loss, data = train_model()
    
    # Create output directory
    os.makedirs('static', exist_ok=True)
    
    # Save model
    model_path = 'static/cifar10_cnn_model.h5'
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Save test data for web app
    test_data = {
        'images': [],
        'labels': [],
        'indices': data['test_indices'].tolist()
    }
    
    # Convert test images to base64 for web app
    for i, (image, label) in enumerate(zip(data['x_test_viz'], data['y_test_viz'])):
        base64_image = array_to_base64_image(image)
        test_data['images'].append(base64_image)
        test_data['labels'].append(int(label))
    
    test_data_path = 'static/test_data.json'
    with open(test_data_path, 'w') as f:
        json.dump(test_data, f)
    print(f"Test data saved to: {test_data_path}")
    
    # Save training plots
    plot_path = save_training_plots(history)
    
    # Save model info
    info_path = save_model_info(model, history, test_accuracy, test_loss)
    
    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print("\nFiles created:")
    print(f"  - Model: {model_path}")
    print(f"  - Test data: {test_data_path}")
    print(f"  - Training plots: {plot_path}")
    print(f"  - Model info: {info_path}")
    print("\nYou can now run the web app with the pre-trained model!")
    print("Run: python app.py")

if __name__ == "__main__":
    main()

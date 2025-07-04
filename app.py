from flask import Flask, request, render_template, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import json
import os

app = Flask(__name__)

# CIFAR-10 class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Global variables for pre-trained model and test data
model = None
test_data = None
model_info = None

def load_pretrained_model():
    """Load the pre-trained model from static directory"""
    global model, model_info
    try:
        model_path = os.path.join('static', 'cifar10_cnn_model.h5')
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return False
            
        print("Loading pre-trained CIFAR-10 CNN model...")
        model = tf.keras.models.load_model(model_path)
        print("Pre-trained model loaded successfully!")
        
        # Load model info
        model_info_path = os.path.join('static', 'model_info.json')
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
            print("Model info loaded successfully!")
        else:
            print("Warning: Model info file not found")
            model_info = {}
            
        return True
    except Exception as e:
        print(f"Error loading pre-trained model: {e}")
        return False

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

def load_test_data():
    """Load test data from static directory with fallback to dummy data"""
    global test_data
    
    try:
        test_data_path = os.path.join('static', 'test_data.json')
        print(f"Trying to load test data from: {test_data_path}")
        
        if not os.path.exists(test_data_path):
            print("Test data file not found, generating dummy data...")
            test_data = generate_dummy_test_data()
            return True
            
        file_size = os.path.getsize(test_data_path)
        print(f"Test data file found. Size: {file_size} bytes ({file_size/(1024*1024):.1f} MB)")
        
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        
        print(f"✅ Test data loaded successfully! {len(test_data['images'])} test images available.")
        return True
        
    except Exception as e:
        print(f"Error loading test data: {e}")
        print("Generating dummy test data as fallback...")
        try:
            test_data = generate_dummy_test_data()
            return True
        except Exception as e2:
            print(f"Failed to generate dummy test data: {e2}")
            return False

def generate_dummy_test_data():
    """Generate dummy CIFAR-10 test data in memory as fallback"""
    print("Generating dummy CIFAR-10 test data...")
    
    dummy_data = {
        'images': [],
        'labels': [],
        'indices': list(range(10))
    }
    
    # Generate 10 dummy 32x32 RGB images (one for each class)
    for i in range(10):
        # Create a simple pattern for each class
        dummy_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        
        # Add some class-specific pattern
        if i == 0:  # airplane - blue-ish
            dummy_image[:, :, 2] = np.minimum(dummy_image[:, :, 2] + 100, 255)
        elif i == 1:  # automobile - gray-ish
            dummy_image = np.mean(dummy_image, axis=2, keepdims=True).repeat(3, axis=2).astype(np.uint8)
        # Add more patterns if needed
        
        base64_image = array_to_base64_image(dummy_image)
        dummy_data['images'].append(base64_image)
        dummy_data['labels'].append(i)
    
    print(f"✅ Generated {len(dummy_data['images'])} dummy test images!")
    return dummy_data

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files including training plots"""
    return send_from_directory('static', filename)

@app.route('/')
def index():
    """Main page"""
    try:
        with open('templates/index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return jsonify({
            "error": "index.html not found",
            "message": "Make sure templates/index.html exists"
        }), 404

@app.route('/api')
def api_info():
    """API information endpoint"""
    return jsonify({
        "message": "CIFAR-10 CNN Classifier - Pre-trained Model Deployment",
        "endpoints": {
            "/api/dataset/info": "Get dataset information",
            "/api/model/info": "Get pre-trained model information",
            "/api/training/plots": "Get training plot image URL",
            "/api/test/random": "Get random test image",
            "/api/test/predict/<index>": "Predict test image label"
        },
        "model_loaded": model is not None,
        "test_data_loaded": test_data is not None,
        "classes": CLASS_NAMES
    })

@app.route('/api/dataset/info')
def dataset_info():
    """Get information about the CIFAR-10 dataset"""
    if test_data is None:
        return jsonify({"error": "Test dataset not loaded"}), 500
    
    return jsonify({
        "test_samples": len(test_data['images']),
        "input_shape": [32, 32, 3],
        "image_shape": [32, 32],
        "num_classes": 10,
        "class_names": CLASS_NAMES,
        "note": "CIFAR-10 dataset with pre-saved test data"
    })

@app.route('/api/model/info')
def model_info_endpoint():
    """Get pre-trained model information"""
    if model is None:
        return jsonify({"error": "Pre-trained model not loaded"}), 500
    
    # Get model summary
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    
    # Count parameters
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    
    response_data = {
        "model_summary": '\n'.join(model_summary),
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "layers": [
            {
                "name": layer.name,
                "type": layer.__class__.__name__,
                "output_shape": str(layer.output.shape) if hasattr(layer, 'output') and layer.output is not None else "Not built"
            }
            for layer in model.layers
        ]
    }
    
    # Add training info if available
    if model_info:
        response_data.update({
            "test_accuracy": model_info.get("test_accuracy"),
            "test_loss": model_info.get("test_loss"),
            "training_info": model_info.get("training_info", {}),
            "note": "Pre-trained CNN model loaded from local training"
        })
    
    return jsonify(response_data)

@app.route('/api/training/plots')
def training_plots():
    """Get the URL for the static training plots image"""
    plot_path = os.path.join('static', 'training_history.png')
    if os.path.exists(plot_path):
        return jsonify({
            "plot_url": "/static/training_history.png",
            "available": True,
            "note": "Static training plots from local CNN training"
        })
    else:
        return jsonify({
            "plot_url": None,
            "available": False,
            "error": "Training plots not found"
        }), 404

@app.route('/api/test/random')
def get_random_test_image():
    """Get a random test image for evaluation"""
    try:
        # Debug information
        print(f"Test data status: {test_data is not None}")
        if test_data is not None:
            print(f"Test data images count: {len(test_data.get('images', []))}")
        
        if test_data is None:
            print("ERROR: Test dataset not loaded")
            return jsonify({
                "error": "Test dataset not loaded",
                "debug": "test_data is None",
                "suggestion": "Check if static/test_data.json exists and is readable"
            }), 500
        
        if 'images' not in test_data or 'labels' not in test_data:
            print("ERROR: Test data missing required keys")
            return jsonify({
                "error": "Test data format invalid",
                "debug": f"Available keys: {list(test_data.keys())}",
                "suggestion": "test_data.json may be corrupted"
            }), 500
        
        if len(test_data['images']) == 0:
            print("ERROR: No test images available")
            return jsonify({
                "error": "No test images available",
                "debug": "images array is empty"
            }), 500
        
        # Get random index
        max_index = len(test_data['images']) - 1
        index = np.random.randint(0, max_index + 1)
        
        print(f"Serving test image index: {index}")
        
        return jsonify({
            "index": int(index),
            "image": test_data['images'][index],
            "true_label": CLASS_NAMES[test_data['labels'][index]],
            "true_label_index": int(test_data['labels'][index])
        })
        
    except Exception as e:
        print(f"ERROR in get_random_test_image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Internal server error",
            "debug": str(e),
            "suggestion": "Check server logs for details"
        }), 500

@app.route('/api/test/predict/<int:index>')
def predict_test_image(index):
    """Predict label for a test image using the pre-trained model"""
    if model is None:
        return jsonify({"error": "Pre-trained model not loaded"}), 500
    
    if test_data is None:
        return jsonify({"error": "Test dataset not loaded"}), 500
    
    if index < 0 or index >= len(test_data['images']):
        return jsonify({"error": "Invalid image index"}), 400
    
    try:
        # Get the image (it's stored as base64, need to decode and preprocess)
        image_data = test_data['images'][index]
        
        # Decode base64 image
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess for model
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {
                'class': CLASS_NAMES[idx],
                'confidence': float(predictions[0][idx])
            }
            for idx in top_3_indices
        ]
        
        return jsonify({
            "predicted_class": CLASS_NAMES[predicted_class_idx],
            "predicted_class_index": int(predicted_class_idx),
            "confidence": float(predictions[0][predicted_class_idx]),
            "true_class": CLASS_NAMES[test_data['labels'][index]],
            "true_class_index": int(test_data['labels'][index]),
            "correct": bool(predicted_class_idx == test_data['labels'][index]),
            "top_3_predictions": top_3_predictions,
            "all_probabilities": [float(p) for p in predictions[0]]
        })
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
def preprocess_image(image):
    """Preprocess image for CIFAR-10 CNN model"""
    # Resize to 32x32 (CIFAR-10 dimensions)
    image = image.resize((32, 32))
    
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array and normalize
    image_array = np.array(image, dtype=np.float32) / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def load_assets_with_retry(max_retries=3):
    """Load assets with retry logic for deployment reliability"""
    import time
    
    for attempt in range(max_retries):
        print(f"Asset loading attempt {attempt + 1}/{max_retries}")
        
        # Load pre-trained model
        model_loaded = load_pretrained_model()
        
        # Load test data
        test_data_loaded = load_test_data()
        
        print(f"Attempt {attempt + 1} results:")
        print(f"  - Model loaded: {model_loaded}")
        print(f"  - Test data loaded: {test_data_loaded}")
        
        if model_loaded and test_data_loaded:
            print("✅ All assets loaded successfully!")
            return True, True
        elif attempt < max_retries - 1:
            print(f"⚠️ Retrying in 2 seconds... (attempt {attempt + 1}/{max_retries})")
            time.sleep(2)
        else:
            print("⚠️ Max retries reached, starting server with available assets")
            
    return model_loaded, test_data_loaded

@app.route('/api/debug')
def debug_status():
    """Debug endpoint to check app status on Railway"""
    import os
    debug_info = {
        "app_status": "running",
        "model_loaded": model is not None,
        "test_data_loaded": test_data is not None,
        "current_directory": os.getcwd(),
        "static_directory_exists": os.path.exists('static'),
        "files_in_static": [],
        "environment": {
            "PORT": os.environ.get('PORT', 'not set'),
            "RAILWAY_ENVIRONMENT": os.environ.get('RAILWAY_ENVIRONMENT', 'not set')
        }
    }
    
    # Check static directory contents
    try:
        if os.path.exists('static'):
            debug_info["files_in_static"] = os.listdir('static')
            
            # Check specific files
            files_to_check = ['cifar10_cnn_model.h5', 'test_data.json', 'model_info.json', 'training_history.png']
            file_status = {}
            for file in files_to_check:
                filepath = os.path.join('static', file)
                if os.path.exists(filepath):
                    file_status[file] = {
                        "exists": True,
                        "size": os.path.getsize(filepath)
                    }
                else:
                    file_status[file] = {"exists": False}
            debug_info["file_status"] = file_status
    except Exception as e:
        debug_info["static_directory_error"] = str(e)
    
    # Test data info
    if test_data is not None:
        debug_info["test_data_info"] = {
            "keys": list(test_data.keys()),
            "images_count": len(test_data.get('images', [])),
            "labels_count": len(test_data.get('labels', []))
        }
    
    return jsonify(debug_info)

@app.route('/api/reload-assets', methods=['POST'])
def reload_assets_manual():
    """Manually reload assets - useful for Railway cold starts"""
    try:
        print("Manual asset reload requested...")
        
        # Force reload both model and test data
        model_loaded, test_data_loaded = load_assets_with_retry(max_retries=3)
        
        result = {
            "success": True,
            "model_loaded": model_loaded,
            "test_data_loaded": test_data_loaded,
            "message": "Asset reload completed"
        }
        
        if model_loaded and test_data_loaded:
            result["message"] = "✅ All assets loaded successfully!"
            result["status"] = "ready"
        elif model_loaded:
            result["message"] = "⚠️ Model loaded but test data failed"
            result["status"] = "partial"
        elif test_data_loaded:
            result["message"] = "⚠️ Test data loaded but model failed"
            result["status"] = "partial"
        else:
            result["message"] = "❌ Both model and test data failed to load"
            result["status"] = "failed"
            result["success"] = False
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error during manual asset reload: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Asset reload failed"
        }), 500

if __name__ == '__main__':
    print("Starting CIFAR-10 CNN Classifier Server...")
    
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    print(f"Server will run on port: {port}")
    
    # Load assets with retry logic
    model_loaded, test_data_loaded = load_assets_with_retry(max_retries=3)
    
    print(f"Final asset loading status:")
    print(f"  - Model loaded: {model_loaded}")
    print(f"  - Test data loaded: {test_data_loaded}")
    
    if model_loaded and test_data_loaded:
        print("✅ All assets loaded successfully! Server ready for full functionality!")
    elif model_loaded:
        print("⚠️ Model loaded but test data failed - server has limited functionality")
    elif test_data_loaded:
        print("⚠️ Test data loaded but model failed - server has limited functionality")
    else:
        print("❌ Both model and test data failed to load - server running with dummy data")
    
    print(f"Starting Flask app on 0.0.0.0:{port}")
    app.run(debug=False, host='0.0.0.0', port=port)

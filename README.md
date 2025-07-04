# CIFAR-10 CNN Classification Web App

A complete machine learning project that trains a Convolutional Neural Network (CNN) on the CIFAR-10 dataset and deploys it as a web application for real-time image classification.

## 🚀 **Live Demo Deployment (Railway)**

This app is perfectly optimized for Railway deployment with the following features:
- ✅ **Pre-trained CNN Model**: No training required - uses a high-performance pre-trained model
- ✅ **Static Assets**: Training plots and model info generated locally for fast loading
- ✅ **Memory Efficient**: Optimized resource usage with pre-saved test data
- ✅ **Instant Testing**: Immediate predictions on CIFAR-10 images
- ✅ **Educational Ready**: Perfect for demonstrating CNN capabilities

### **Quick Railway Deployment:**

1. **Push to GitHub** (if not already done):
   ```bash
   git add .
   git commit -m "Ready for Railway deployment with pre-trained CNN model"
   git push origin main
   ```

2. **Deploy to Railway**:
   - Go to [railway.app](https://railway.app)
   - Sign up with your GitHub account
   - Click "New Project" → "Deploy from GitHub repo"
   - Select this repository
   - Railway will automatically detect the Python app and deploy!

3. **Automatic Setup**:
   - Railway reads `Procfile` and `requirements.txt`
   - Loads pre-trained model (`static/cifar10_cnn_model.h5`)
   - Serves static training plots and assets
   - Assigns a public URL (e.g., `https://cifar10-cnn-production.up.railway.app`)

4. **Share with Students**:
   - Copy the Railway URL and share with your class
   - Students can immediately test the CNN on CIFAR-10 images
   - Perfect for demonstrating machine learning concepts!

## 🚀 Features

- **CNN Training**: Deep learning model trained on CIFAR-10 dataset
- **Web Interface**: Beautiful, responsive web app for image classification
- **Real-time Predictions**: Instant classification with confidence scores
- **Top-10 Class Probabilities**: Shows all CIFAR-10 classes with confidence bars
- **Educational Interface**: Model summary, architecture details, and training history

## 📊 CIFAR-10 Dataset

The model classifies images into 10 categories:
- ✈️ Airplane
- 🚗 Automobile  
- 🐦 Bird
- 🐱 Cat
- 🦌 Deer
- 🐕 Dog
- 🐸 Frog
- 🐴 Horse
- 🚢 Ship
- 🚛 Truck

## 🛠️ Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the CNN model (optional - pre-trained model included):**
   ```bash
   python train_cnn.py
   ```

3. **Run the web application:**
   ```bash
   python app.py
   ```
   The web app will be available at `http://localhost:5000`

## 🏗️ Model Architecture

The CNN includes:
- 3 Convolutional blocks with BatchNormalization and Dropout
- MaxPooling layers for spatial reduction
- Dense layers with regularization
- Data augmentation for better generalization

Key features:
- Input: 32x32x3 RGB images
- Output: 10 classes (softmax activation)
- Optimizer: Adam
- Loss: Categorical crossentropy

## 📁 Project Structure

```
cnn_classification/
├── app.py                    # Flask web application
├── train_cnn.py             # CNN training script (optional)
├── requirements.txt         # Python dependencies
├── Procfile                 # Railway deployment config
├── railway.toml            # Railway optimization settings
├── templates/
│   └── index.html          # Web interface
├── static/                 # Pre-trained assets (Railway ready)
│   ├── cifar10_cnn_model.h5   # Trained CNN model
│   ├── test_data.json         # Test images for demo
│   ├── training_history.png   # Training plots
│   └── model_info.json        # Model metadata
└── README.md               # This file
```

## 🎨 Web Interface

The web application features:
- Modern, responsive design following educational best practices
- Real-time predictions with confidence scores
- Interactive testing on real CIFAR-10 images
- Model architecture and training history display
- Mobile-friendly interface

## 🔧 Technical Details

- **Framework**: TensorFlow/Keras for deep learning
- **Web Framework**: Flask for the web application
- **Frontend**: HTML5, CSS3, JavaScript (vanilla)
- **Image Processing**: PIL (Pillow) for image preprocessing
- **Visualization**: Matplotlib for training plots
- **Deployment**: Railway-optimized with Gunicorn

## 📈 Model Performance

The CNN achieves competitive accuracy on CIFAR-10 through:
- Data augmentation (rotation, shifts, flips, zoom)
- Batch normalization for stable training
- Dropout for regularization
- Early stopping to prevent overfitting
- Learning rate reduction on plateau

## 🎓 Educational Use

Perfect for:
- Machine learning courses
- CNN architecture demonstrations  
- Understanding computer vision concepts
- Interactive exploration of deep learning
- Real-time image classification demos
- Hands-on prediction testing

## API Endpoints

- `GET /` - Main application interface
- `GET /api` - API information and health check
- `GET /api/dataset/info` - CIFAR-10 dataset statistics
- `GET /api/model/info` - CNN model architecture information
- `GET /api/training/plots` - Training history plots
- `GET /api/test/random` - Get random test image
- `GET /api/test/predict/<index>` - Predict test image label

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

## 📄 License

This project is open source and available under the MIT License.
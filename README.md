
# VisionNet: CIFAR-10 Image Classifier Using Convolutional Neural Networks

## **Project Overview**
VisionNet is an advanced tool for training, evaluating, and experimenting with Convolutional Neural Networks (CNNs) for image classification. This project uses the CIFAR-10 dataset, a standard benchmark dataset containing 60,000 32x32 color images from 10 classes, such as airplanes, cars, and cats. VisionNet incorporates data augmentation, feature visualization, and architectural experimentation.

---

## **Problem Statement**
The goal of VisionNet is to:
1. Build a CNN for image classification.
2. Improve performance using data augmentation techniques.
3. Visualize feature maps from convolutional layers to understand how CNNs extract features.
4. Experiment with different CNN architectures to compare performance.

---

## **Project Features**
- **Loading and preprocessing the CIFAR-10 dataset.**
- **CNN implementation using TensorFlow/Keras.**
- **Data augmentation for performance improvement.**
- **Visualization of feature maps from convolutional layers.**
- **Experimentation with architectures like LeNet.**

---

## **Concepts and Definitions**

### **What is a Convolutional Neural Network (CNN)?**
A CNN is a specialized neural network designed for processing structured grid data such as images. It automates feature extraction by learning patterns like edges, textures, and shapes from raw image data.

### **Key Components of CNNs**
1. **Convolution Layer**: Applies filters (kernels) to input images to extract feature maps.
2. **Pooling Layer**: Reduces spatial dimensions of feature maps, preserving significant features.
3. **Activation Functions**: Non-linear functions like ReLU are used to add non-linearity to the network.
4. **Fully Connected Layer**: Combines extracted features for classification tasks.

### **Data Augmentation**
Data augmentation artificially expands the training dataset by applying transformations like rotations, flips, and shifts. This improves generalization and prevents overfitting.

### **Feature Maps**
Feature maps represent the output of convolution layers. They highlight regions of the image where specific features (e.g., edges or patterns) are detected.

### **LeNet Architecture**
LeNet is one of the earliest CNN architectures designed for handwritten digit recognition. It consists of convolutional layers, average pooling layers, and dense layers.

---

## **Steps and Workflow**

### **1. Import Libraries**
The required libraries include:
- **TensorFlow/Keras** for building CNN models.
- **Matplotlib** for visualization.
- **NumPy** for numerical operations.

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
```

---

### **2. Load and Preprocess Data**
- Load CIFAR-10 dataset.
- Normalize pixel values to the range [0, 1].
- Flatten labels for compatibility with models.

```python
def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train, y_test = y_train.flatten(), y_test.flatten()
    return x_train, y_train, x_test, y_test
```

---

### **3. Define CNN Architecture**
The CNN consists of:
1. Three convolutional layers with ReLU activations.
2. Two max-pooling layers for down-sampling.
3. A dense output layer with a softmax activation for classification.

```python
def create_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model
```

---

### **4. Compile the Model**
Compilation involves:
- **Loss Function**: `sparse_categorical_crossentropy` for multi-class classification.
- **Optimizer**: Adam optimizer for faster convergence.
- **Metrics**: Accuracy.

```python
def compile_model(model):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
```

---

### **5. Train the Model**
Train the model using the training dataset for a specified number of epochs.

```python
def train_model(model, x_train, y_train, x_test, y_test, epochs=10):
    history = model.fit(x_train, y_train, epochs=epochs,
                        validation_data=(x_test, y_test))
    return history
```

---

### **6. Evaluate Model Performance**
Evaluate the trained model using test data to calculate accuracy and loss.

```python
def evaluate_model(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}")
    return test_acc
```

---

### **7. Apply Data Augmentation**
Enhance the dataset with augmented images to improve model robustness.

```python
def apply_data_augmentation(x_train, y_train, x_test, y_test):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    datagen.fit(x_train)
    augmented_model = create_cnn_model()
    augmented_model = compile_model(augmented_model)
    augmented_model.fit(datagen.flow(x_train, y_train, batch_size=32),
                        epochs=10, validation_data=(x_test, y_test))
    return augmented_model
```

---

### **8. Visualize Feature Maps**
Visualize intermediate feature maps to understand how the model extracts features.

```python
def visualize_feature_maps(model, x_sample):
    layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(x_sample[np.newaxis, ...])
    for i, activation in enumerate(activations):
        fig, ax = plt.subplots(4, 8, figsize=(12, 6))
        fig.suptitle(f"Feature Maps from Layer {i + 1}")
        for j in range(activation.shape[-1]):
            ax[j // 8, j % 8].imshow(activation[0, :, :, j], cmap='viridis')
            ax[j // 8, j % 8].axis('off')
        plt.show()
```

---

### **9. Experiment with Different Architectures**
Experiment with classic architectures like LeNet.

```python
def create_lenet_model():
    model = models.Sequential([
        layers.Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 3)),
        layers.AveragePooling2D(),
        layers.Conv2D(16, (5, 5), activation='relu'),
        layers.AveragePooling2D(),
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model
```

---

## **How to Run the Project**
1. Install dependencies:
   ```bash
   pip install tensorflow numpy matplotlib
   ```
2. Execute the Python script:
   ```bash
   python VisionNet_CNN_Classifier.py
   ```
3. Follow the console output to track progress.

---

## **Results and Observations**
- Initial accuracy with CNN: ~70%.
- Enhanced accuracy with data augmentation: ~75-80%.
- Feature map visualizations reveal patterns learned by the model.

---

## **Future Improvements**
- Explore deeper architectures like ResNet.
- Use transfer learning for better performance on small datasets.
- Experiment with hyperparameter tuning.

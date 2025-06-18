# 🌿 Plant Disease Detection using CNN

This project aims to build a **Convolutional Neural Network (CNN)** model to automatically detect plant leaf diseases from images. Early and accurate identification of plant diseases can help farmers take timely action to protect crops and increase yield.

## 📂 Dataset

The dataset used in this project is from Kaggle:  
👉 [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)

- The dataset contains **over 50,000+ images** of healthy and diseased plant leaves.
- Covers **14 crop species** and **38 different classes** (diseased and healthy).
- Images are already **preprocessed and labeled** in subdirectories.

## 🧠 Objective

To train a CNN model that can classify leaf images into their corresponding disease classes and help in **automated plant health monitoring**.

## 📌 Technologies Used

- Python
- TensorFlow / Keras
- NumPy, Pandas
- OpenCV (for image preprocessing)
- Matplotlib, Seaborn (for visualization)

## 🧪 Model Overview

- **Model Type:** Sequential CNN  
- **Input Shape:** Resized images  
- **Layers:** Conv2D, MaxPooling2D, Dropout, Dense  
- **Activation:** ReLU, Softmax  
- **Loss Function:** Categorical Crossentropy  
- **Optimizer:** Adam

## 📊 Results

- Achieved high accuracy on training and validation data
- Confusion matrix and classification reports show strong performance across most classes


## 📁 Project Structure


## 🚀 Future Improvements

- Integrate into a **mobile app** for real-time disease detection in the field.
- Use **transfer learning** (e.g., with EfficientNet or ResNet).
- Deploy model using **Flask** or **Streamlit** for demo.

## 🧑‍🏫 Credits

- Dataset: [PlantVillage on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- Guidance and support from open-source communities and academic references.

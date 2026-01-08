👁️ Eye Disease Classification using EfficientNetB3

This repository features a Deep Learning solution for the automated detection and classification of eye diseases from retinal images. By leveraging Transfer Learning with the EfficientNetB3 architecture, the model achieves high diagnostic accuracy, which can serve as a supportive tool for ophthalmologists.

🌟 Key Features

State-of-the-Art Architecture: Utilizes EfficientNetB3 pre-trained on ImageNet for robust feature extraction.

Multi-Class Classification: Designed to identify multiple conditions, typically including:

Cataract

Diabetic Retinopathy

Glaucoma

Normal

Advanced Image Preprocessing: Includes image resizing, normalization, and data augmentation to improve model generalization.

Optimized Training: Implements callbacks such as EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint to ensure efficient training and save the best-performing model.

GPU Accelerated: Configured to utilize TensorFlow with GPU acceleration for faster processing.

🛠 Project Structure and Files
<img width="1017" height="206" alt="image" src="https://github.com/user-attachments/assets/c368c3a9-c367-4a78-ad0e-3871eb0911fd" />

🚀 How to Run the Project

1. Setup and Installation

# Clone the repository
git clone [https://github.com/ayush1k/Eye-Disease-Classification-EfficientNetB3.git](https://github.com/ayush1k/Eye-Disease-Classification-EfficientNetB3.git)
cd Eye-Disease-Classification-EfficientNetB3

# Install dependencies
pip install -r requirements.txt


2. Dataset Preparation

The notebook uses a structured image dataset. Ensure your data directory is set up as follows:

dataset/
    cataract/
    diabetic_retinopathy/
    glaucoma/
    normal/


Note: Update the Path variables in the notebook to point to your local dataset location.

3. Execution

Launch the notebook environment:

jupyter notebook eye_disease.ipynb


💡 Technologies Used

Framework: TensorFlow / Keras

Model: EfficientNetB3

Languages: Python

Libraries: NumPy, Pandas, Matplotlib, Seaborn, OpenCV (cv2)

Environment: Google Colab / Kaggle (GPU enabled)

📊 Training Performance

The model was trained for 20 epochs. The final training and validation metrics are summarized below:
<img width="988" height="143" alt="image" src="https://github.com/user-attachments/assets/a495aee4-a5be-431a-a782-c61253e4ab28" />
The model demonstrates high precision across all classes, with the EarlyStopping callback ensuring the model saved is the one with the best validation performance.

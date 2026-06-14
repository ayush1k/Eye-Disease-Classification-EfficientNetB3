# 👁️ Eye Disease Classification using EfficientNetB3

This repository features a Deep Learning solution for the automated detection and classification of eye diseases from retinal images. By leveraging Transfer Learning with the EfficientNetB3 architecture, the model achieves high diagnostic accuracy, which can serve as a supportive tool for ophthalmologists.

---

## 🌟 Key Features

* **State-of-the-Art Architecture**: Utilizes EfficientNetB3 pre-trained on ImageNet for robust feature extraction.
* **Multi-Class Classification**: Designed to identify multiple conditions, including:
  * Cataract
  * Diabetic Retinopathy
  * Glaucoma
  * Normal (Healthy)
* **Advanced Image Preprocessing**: Includes image resizing, normalization, and data augmentation to improve model generalization.
* **Optimized Training**: Implements callbacks such as `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint` to ensure efficient training and save the best-performing model.
* **GPU Accelerated**: Configured to utilize TensorFlow with GPU acceleration for faster processing.

---

## 🏗️ System Architecture & Mechanics

This section details the architectural blueprint and computational design of the classification pipeline defined in [eye_disease (1).ipynb](file:///workspaces/Eye-Disease-Classification-EfficientNetB3/eye_disease%20(1).ipynb).

### 1. The Executive Blueprint
The codebase is designed to build a deep learning pipeline utilizing transfer learning for ophthalmic diagnostics.

* **Core Language**: Python 3
* **Framework**: TensorFlow & Keras
* **Primary Backbone**: `EfficientNetB3` (pre-trained on ImageNet)
* **Libraries**: NumPy, Pandas, OpenCV (`cv2`), Pillow (`PIL`), Scikit-Learn (for model selection and metrics evaluation), Matplotlib, and Seaborn.

### 2. Execution Entry Points & Lifecycle
The execution is structured linearly within [eye_disease (1).ipynb](file:///workspaces/Eye-Disease-Classification-EfficientNetB3/eye_disease%20(1).ipynb):
1. **Environment Setup & Mounting**: Connects Google Drive to load source directories and install visualization components.
2. **Metadata Framing**: Instantiates `main_class` to scan folders recursively (`**/*.*`) and builds a Pandas DataFrame mapping image paths to encoded class indices.
3. **Data Prep & Model Definition**: Invokes the `EfficientNet_call` method which loads, resizes, and shuffles data, splits it (90/10 train-test), assembles the custom Keras network graph, and triggers training.
4. **Validation & Evaluation**: Runs validation inference, generating confusion matrices, classification reports, and accuracy graphs.

### 3. Data Flow Pipeline
The end-to-end data flow operates through the following states:

```mermaid
flowchart LR
    Disk["Raw Disk Data (.jpg/.png)"]
    DF["Pandas DataFrame (Link, Label, Name)"]
    RAM["In-Memory NumPy Arrays (X_train, Y_train)"]
    Model["EfficientNetB3 Neural Net Model"]
    Outputs["Metrics, Plotting, and saved efficient.h5"]

    Disk -- "pathlib scanning" --> DF
    DF -- "cv2.imread & cv2.resize" --> RAM
    RAM -- "model.fit() & batching" --> Model
    Model -- "predictions" --> Outputs
```

* **Custom Classification Head**:
  The pre-trained CNN layers serve as feature extractors. The output is fed into a custom classifier:
  $$\text{EfficientNetB3 Output} \longrightarrow \text{GlobalAveragePooling2D} \longrightarrow \text{Dropout (0.5)} \longrightarrow \text{Dense (4 Neurons, Softmax)}$$
* **Observers/Callbacks**:
  * `ReduceLROnPlateau` decays the learning rate by $0.3\times$ on accuracy plateaus.
  * `EarlyStopping` terminates training early if validation loss fails to improve for 5 consecutive epochs.
  * `ModelCheckpoint` saves the highest performing model to `efficient.h5`.

---

## ⚠️ Critical Gotchas & Architectural Review

Before you modify or train the model, you should be aware of several high-priority code complexities and anomalies:

### 1. Memory Scalability (Out-of-Memory Risk)
* **Gotcha**: The codebase loads the entire dataset into RAM at once using a python loop over `cv2.imread`.
* **Impact**: For larger clinical datasets, this will exhaust system memory and result in an OOM crash.
* **Recommendation**: Refactor the data pipeline to stream files using `tf.data.Dataset` or a custom Keras Generator.

### 2. Inconsistent Loss and Activation Functions
* **Gotcha**: The network output uses a `softmax` activation (probability-based) but compiles using the `squared_hinge` loss function.
* **Impact**: Hinge loss is typically paired with margin-based outputs. Softmax paired with Squared Hinge leads to suboptimal gradient calculations and slower convergence.
* **Recommendation**: Compile the model with `loss='categorical_crossentropy'`.

### 3. Image Color Channel Discrepancies
* **Gotcha**: `img_show` loads images in grayscale via PIL, but the training function loads images in BGR format via `cv2.imread`. 
* **Impact**: Pre-trained `EfficientNetB3` models expect standard RGB images. Training on BGR without conversion degrades classification performance at test time.
* **Recommendation**: Standardize the pipeline to read and convert images to RGB (`cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`).

### 4. Recursive Globbing and Split Hygiene
* **Gotcha**: The workspace preparation shell script copies files into both `/content/dataset/Testing` and `/content/dataset/Training`, but the Python loader reads the parent folder `/content/dataset` recursively.
* **Impact**: Every image is loaded twice (once as testing and once as training), which duplicates records in memory and invalidates test-train split validation results.
* **Recommendation**: Adjust the Python loader to read only from `/content/dataset/Training/`.

---

## 🛠️ Project Structure and Files
<img width="1017" height="206" alt="image" src="https://github.com/user-attachments/assets/c368c3a9-c367-4a78-ad0e-3871eb0911fd" />

---

## 🚀 How to Run the Project

### 1. Setup and Installation
```bash
# Clone the repository
git clone https://github.com/ayush1k/Eye-Disease-Classification-EfficientNetB3.git
cd Eye-Disease-Classification-EfficientNetB3

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation
Ensure your data directory is set up as follows:
```text
dataset/
    cataract/
    diabetic_retinopathy/
    glaucoma/
    normal/
```
*Note: Update the path variables in the notebook to point to your local dataset location.*

### 3. Execution
Launch the notebook environment:
```bash
jupyter notebook eye_disease.ipynb
```

---

## 📊 Training Performance

The model was trained for 20 epochs. The final training and validation metrics are summarized below:

<img width="988" height="143" alt="image" src="https://github.com/user-attachments/assets/a495aee4-a5be-431a-a782-c61253e4ab28" />

The model demonstrates high precision across all classes, with the `EarlyStopping` callback ensuring the model saved is the one with the best validation performance.

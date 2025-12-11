# 🌸 Task-3: Iris Flower Classification  
CodSoft Data Science Internship — Task 3

This project builds a Machine Learning model to classify **Iris flower species** based on four input features using the KNN algorithm. The project includes data preprocessing, visualization, model training, evaluation, and saving output files.

---

## 📂 Dataset
The dataset used is the classic **Iris dataset** containing:

- 150 samples  
- 4 features:  
  - sepal_length  
  - sepal_width  
  - petal_length  
  - petal_width  
- 1 target label: species  

Dataset location:
data/IRIS.csv

---

## 🚀 Project Workflow

### **1️⃣ Load Dataset**
- Load CSV  
- Print head  
- Drop unnecessary columns if any  

### **2️⃣ Data Visualization**
Generated and saved inside `output_screenshots/`:
- Species Distribution Plot  
- Pairplot  
- Confusion Matrix  

### **3️⃣ Preprocessing**
- Separate features (X) and labels (y)
- Standardize numerical features using **StandardScaler**
- Train/Test split: 80/20

### **4️⃣ Model Training**
Model used:
- **KNN Classifier** (`n_neighbors = 5`)

### **5️⃣ Model Evaluation**
Metrics:
- Accuracy score  
- Classification report  
- Confusion matrix heatmap  

---

## 📁 Project Structure

Task-3/
│── data/
│ └── IRIS.csv
│── models/
│ ├── knn_model.pkl
│ └── scaler.pkl
│── output_screenshots/
│ ├── confusion_matrix.png
│ ├── pairplot.png
│ └── species_distribution.png
│── src/
│ └── iris_classification.py
│── README.md
└── requirements.txt

---

## 🧠 How to Run the Project

### **1️⃣ Create Virtual Environment**
python -m venv .venv


### **2️⃣ Activate Environment**  
Windows:


.venv\Scripts\activate

### **3️⃣ Install Dependencies**
pip install -r requirements.txt


### **4️⃣ Run Script**


python src/iris_classification.py


---

## 📊 Output Files

### Models saved:


models/knn_model.pkl
models/scaler.pkl
### Graphs saved:
output_screenshots/confusion_matrix.png
output_screenshots/pairplot.png
output_screenshots/species_distribution.png

yaml
Copy code

---

## 🏁 Conclusion
This project demonstrates:
- Data preprocessing  
- Data visualization  
- Machine learning model training  
- Model evaluation  
- Saving trained models and visual outputs  

A complete end-to-end ML project for classification tasks.

---

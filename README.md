# Alzheimer's Disease Prediction using Machine Learning

This project is focused on analyzing and predicting Alzheimer's Disease using a structured machine learning pipeline with `RandomForestClassifier`. The dataset is explored, cleaned, and then used to train and evaluate a model.

## 📁 Project Structure

- `ds-project.ipynb`: Jupyter notebook containing all steps of the project.
- `alzheimers_disease_data.csv`: The dataset used.

## 🚀 Project Steps

1. **Data Loading**
   - Load the dataset using `pandas`.

2. **Exploratory Data Analysis (EDA)**
   - Visualize features using `seaborn` and `matplotlib` to identify trends and patterns.
   - Understand feature distributions and correlations.

3. **Data Cleaning**
   - Drop unnecessary or irrelevant columns.
   - Handle missing or inconsistent values.

4. **Data Splitting**
   - Split the dataset into training and test sets (`train` and `check`) for model validation.

5. **Model Training**
   - Train a `RandomForestClassifier` on the prepared data.
   - Initial model performance score: ~60%

6. **Results Visualization**
   - Use a heatmap to compare the model's predictions with actual values.

7. **Model Optimization**
   - Retrain the model with `class_weight='balanced'` to address class imbalance.
   - Improved model performance score: ~92%

8. **Final Evaluation**
   - Re-evaluate the improved model using a heatmap to visualize prediction accuracy.

## 📊 Technologies Used

- Python
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## 🧪 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/tamar2791/Alzheimer-s-AI-Project.git
   ```
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open and run `ds-project.ipynb`.

## 📌 Future Improvements

- Test other classification models (e.g., XGBoost, SVM)
- Perform cross-validation
- Tune hyperparameters using GridSearch or RandomSearch
- Add more robust handling of class imbalance

---


## 📥 Dataset Source

The dataset used in this project was downloaded from [Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset/data).  

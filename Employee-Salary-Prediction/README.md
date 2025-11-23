# Employee Salary Prediction

A machine learning project that predicts employee income levels (>50K or <=50K) based on demographic and employment-related features using various classification algorithms.

## 📋 Overview

This project uses the Adult Census Income dataset to build and compare multiple machine learning models for binary classification. The goal is to predict whether an individual's annual income exceeds $50,000 based on features such as age, education, occupation, work class, and other demographic information.

## ✨ Features

- **Comprehensive Data Preprocessing**: Missing value handling, outlier detection and removal
- **Data Cleaning**: Removal of irrelevant categories and redundant features
- **Feature Engineering**: Label encoding for categorical variables and MinMax scaling
- **Multiple ML Models**: Comparison of 5 different classification algorithms
- **Model Evaluation**: Accuracy scoring and classification reports
- **Visualization**: Boxplots for outlier analysis and bar charts for model comparison
- **Model Persistence**: Automatic saving of the best-performing model

## 🔧 Requirements

This project requires the following Python libraries:

```bash
pandas
numpy
matplotlib
scikit-learn
joblib
```

### Installation

Install all dependencies using pip:

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

## 📊 Dataset

The project uses the **Adult Census Income dataset** (`adult 3.csv`) containing the following features:

### Input Features:
- **age**: Age of the individual
- **workclass**: Employment type (Private, Self-emp, Government, etc.)
- **educational-num**: Number of years of education
- **marital-status**: Marital status
- **occupation**: Type of occupation
- **relationship**: Relationship status
- **race**: Race of the individual
- **gender**: Gender (Male/Female)
- **capital-gain**: Capital gains
- **capital-loss**: Capital losses
- **hours-per-week**: Working hours per week
- **native-country**: Country of origin

### Target Variable:
- **income**: Binary classification (>50K or <=50K)

## 🚀 Usage

1. **Prepare your data**: Ensure the `adult 3.csv` file is in the correct path
2. **Update file path**: Modify `/content/adult 3.csv` in the code if needed
3. **Run the script**: Execute the Python file

```bash
python employeesalaryprediction.py
```

4. **View results**: Check console output for accuracy scores and generated visualizations

## 🧹 Data Preprocessing Steps

### 1. Missing Value Handling
- Replaced `?` in `occupation` with "Not Listed"
- Replaced `?` in `workclass` with "others"

### 2. Data Cleaning
- Removed rows with `workclass` = "Without-pay" or "Never-worked"
- Removed rows with low education levels (Preschool, 1st-4th, 5th-6th)
- Dropped redundant `education` column (kept `educational-num`)

### 3. Outlier Removal
- **Age**: Filtered to range [17, 75]
- **Educational-num**: Filtered to range [5, 16]
- Visualized outliers using boxplots

### 4. Feature Encoding
- Applied **Label Encoding** to categorical features:
  - workclass, marital-status, occupation, relationship, gender, race, native-country
- Applied **MinMax Scaling** to normalize all features to [0, 1] range

## 🤖 Machine Learning Models

The project compares 5 classification algorithms:

1. **Logistic Regression**
2. **Random Forest Classifier**
3. **K-Nearest Neighbors (KNN)**
4. **Support Vector Machine (SVM)**
5. **Gradient Boosting Classifier**

### Model Training Process

- **Train/Test Split**: 80% training, 20% testing (stratified sampling)
- **Pipeline**: StandardScaler + Model
- **Evaluation Metrics**: Accuracy score and classification report
- **Best Model Selection**: Automatically selects and saves the model with highest accuracy

## 📈 Model Evaluation

The script provides:
- **Individual Model Accuracies**: Printed for each algorithm
- **Classification Reports**: Precision, recall, F1-score for each model
- **Visual Comparison**: Bar chart comparing all model accuracies
- **Best Model Identification**: Highlights the top-performing algorithm

### Example Output

```
LogisticRegression: 0.8456
RandomForest: 0.8612
KNN: 0.8234
SVM: 0.8501
GradientBoosting: 0.8678

✅ Best model: GradientBoosting with accuracy 0.8678
✅ Saved best model as best_model.pkl
```

## 💾 Model Persistence

The project automatically saves:
- **best_model.pkl**: The trained model with the highest accuracy
- **encoders.pkl**: Label encoders for categorical features (for future predictions)

### Loading the Saved Model

```python
import joblib

# Load model and encoders
model = joblib.load('best_model.pkl')
encoders = joblib.load('encoders.pkl')

# Make predictions
prediction = model.predict(new_data)
```

## 📊 Visualizations

The project generates:
1. **Boxplots** for outlier detection in numerical features (age, capital-gain, capital-loss, educational-num, hours-per-week)
2. **Bar Chart** comparing accuracy scores of all models

## 🎯 Key Insights

- The model successfully predicts income levels with accuracy typically between 82-87%
- Gradient Boosting and Random Forest tend to perform best on this dataset
- Feature engineering and outlier removal significantly improve model performance
- Stratified sampling ensures balanced representation of both income classes

## 🔄 Customization

You can customize:
- **Train/Test Split Ratio**: Modify `test_size` parameter (default: 0.2)
- **Random State**: Change `random_state` for reproducibility
- **Model Hyperparameters**: Adjust parameters for each classifier
- **Feature Selection**: Add/remove features based on domain knowledge
- **Outlier Thresholds**: Modify age, education ranges based on your data

## 🛠️ Troubleshooting

- **File not found error**: Verify the dataset path is correct
- **Memory issues**: Reduce dataset size or use incremental learning
- **Low accuracy**: Try feature engineering, hyperparameter tuning, or ensemble methods
- **Encoding errors**: Ensure all categorical variables are properly encoded

## 📚 Technical Details

- **Problem Type**: Binary Classification
- **Learning Type**: Supervised Learning
- **Framework**: scikit-learn
- **Preprocessing**: Label Encoding, MinMax Scaling
- **Evaluation Strategy**: Train-Test Split with Stratification

## 👨‍💻 Author

Developed for educational purposes to demonstrate income prediction using machine learning classification algorithms.

## 📄 License

This project is available for educational and research purposes.

---

**Note**: Ensure the dataset path (`/content/adult 3.csv`) is correctly set before running the script. The Adult Census Income dataset is publicly available from the UCI Machine Learning Repository.
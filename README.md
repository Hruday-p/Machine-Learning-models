# Machine Learning Models 🤖

A comprehensive collection of machine learning projects demonstrating various ML techniques including deep learning (LSTM), classification algorithms, and cryptocurrency price prediction using Random Forest. This repository showcases practical implementations of ML models for real-world problems.

## 📋 Overview

This repository contains three major machine learning projects:
1. **Cryptocurrency Price Prediction** - Random Forest regression for Bitcoin, Ethereum, and TRON
2. **Employee Salary Prediction** - Multi-algorithm classification for income prediction
3. **Temperature Forecasting** - LSTM neural network for time series prediction

## 📂 Repository Structure

```
Machine-Learning-models/
│
├── BT/                                      # Blockchain/Cryptocurrency Price Prediction
│   ├── Untitled2.ipynb                      # Main notebook for crypto prediction
│   ├── Bitcoin Historical Data*.csv         # Bitcoin price datasets (3 files)
│   ├── Ethereum Historical Data*.csv        # Ethereum price datasets (3 files)
│   ├── TRON Inc Stock Price History*.csv    # TRON price datasets (3 files)
│   └── README.md                            # Project documentation
│
├── Employee-Salary-Prediction/              # Income Classification
│   ├── employeesalaryprediction.py          # Main Python script
│   ├── adult 3.csv                          # Adult Census Income dataset
│   └── README.md                            # Project documentation
│
└── Temparature-forecasting-using-LSTM/      # Time Series Forecasting
    ├── temperature_forecast_using_lstm.py   # Main Python script
    ├── predictive_maintenance_dataset.csv   # Temperature dataset
    └── README.md                            # Project documentation
```

## 🚀 Projects

### 1. Cryptocurrency Price Prediction (BT/)

**Description:** Predicts cryptocurrency prices for Bitcoin, Ethereum, and TRON using Random Forest regression.

**Key Features:**
- Multi-asset support (3 cryptocurrencies)
- Data concatenation from multiple historical CSV files
- Advanced data cleaning (K/M notation, percentage handling)
- Feature scaling with MinMaxScaler
- Random Forest with 100 estimators
- Performance evaluation using MAE and RMSE

**Technologies:**
- pandas, numpy
- scikit-learn (RandomForestRegressor, MinMaxScaler)
- Jupyter Notebook

**Dataset:**
- 9 CSV files with historical price data (Date, Price, Open, High, Low, Volume, Change %)
- Multiple time periods for each cryptocurrency

**Performance Metrics:**
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

---

### 2. Employee Salary Prediction

**Description:** Binary classification to predict whether an individual's income exceeds $50K based on demographic features.

**Key Features:**
- Comprehensive data preprocessing and cleaning
- Missing value handling and outlier removal
- Label encoding for categorical features
- Comparison of 5 ML algorithms
- Automatic best model selection and saving
- Visualization with boxplots and bar charts

**Technologies:**
- pandas, numpy, matplotlib
- scikit-learn (LogisticRegression, RandomForest, KNN, SVM, GradientBoosting)
- joblib for model persistence

**Dataset:**
- Adult Census Income dataset
- Features: age, workclass, education, occupation, gender, race, hours-per-week, etc.
- Target: income (>50K or <=50K)

**ML Models Compared:**
1. Logistic Regression
2. Random Forest Classifier
3. K-Nearest Neighbors (KNN)
4. Support Vector Machine (SVM)
5. Gradient Boosting Classifier

**Typical Accuracy:** 82-87%

---

### 3. Temperature Forecasting using LSTM

**Description:** Time series forecasting using Long Short-Term Memory (LSTM) neural networks to predict hourly temperature.

**Key Features:**
- Stacked LSTM architecture (2 layers)
- Sequence generation using sliding window (24 timesteps)
- MinMaxScaler normalization
- Real-time prediction and visualization
- RMSE evaluation metric

**Technologies:**
- TensorFlow/Keras (LSTM layers)
- pandas, numpy, matplotlib
- scikit-learn (MinMaxScaler)

**Dataset:**
- Predictive maintenance dataset with temperature readings
- Features: Timestamp, Temperature

**Model Architecture:**
```
Input Layer (24 timesteps, 1 feature)
    ↓
LSTM Layer (50 units, return_sequences=True)
    ↓
LSTM Layer (50 units)
    ↓
Dense Layer (1 unit)
```

**Hyperparameters:**
- Timesteps: 24 hours
- LSTM Units: 50 per layer
- Epochs: 20
- Batch Size: 32
- Train/Test Split: 80/20

---

## 🔧 Installation

### Prerequisites

Ensure you have Python 3.8+ installed.

### Clone the Repository

```bash
git clone https://github.com/Hruday-p/Machine-Learning-models.git
cd Machine-Learning-models
```

### Install Dependencies

#### For Cryptocurrency Price Prediction (BT/)
```bash
pip install pandas numpy scikit-learn jupyter
```

#### For Employee Salary Prediction
```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

#### For Temperature Forecasting
```bash
pip install pandas numpy tensorflow scikit-learn matplotlib
```

### Or Install All Dependencies at Once

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib joblib jupyter
```

## 🎯 Usage

### Cryptocurrency Price Prediction

```bash
cd BT
jupyter notebook Untitled2.ipynb
```

Run all cells to:
1. Load and concatenate historical price data
2. Preprocess and scale features
3. Train Random Forest models
4. Generate predictions and evaluate performance
5. Export results to CSV

### Employee Salary Prediction

```bash
cd Employee-Salary-Prediction
python employeesalaryprediction.py
```

The script will:
1. Load and clean the Adult Census dataset
2. Handle missing values and remove outliers
3. Encode categorical features
4. Train 5 different classification models
5. Compare accuracy scores
6. Save the best model as `best_model.pkl`

### Temperature Forecasting

```bash
cd Temparature-forecasting-using-LSTM
python temperature_forecast_using_lstm.py
```

The script will:
1. Load temperature data
2. Create sequences for supervised learning
3. Train the stacked LSTM model
4. Generate predictions
5. Calculate RMSE
6. Plot actual vs predicted temperatures

## 📊 Results

### Cryptocurrency Price Prediction
- **Bitcoin**: MAE ~$829, RMSE ~$994
- **Ethereum**: MAE ~$127, RMSE ~$157
- **TRON**: MAE ~$0.75, RMSE ~$0.98

### Employee Salary Prediction
- Best model accuracy: **86-87%** (typically Gradient Boosting or Random Forest)
- Precision, Recall, F1-score available in classification reports

### Temperature Forecasting
- RMSE varies based on dataset and temperature range
- Visual comparison shows strong correlation between actual and predicted values

## 🛠️ Technologies Used

### Programming Languages
- Python 3.8+

### Machine Learning Frameworks
- **scikit-learn**: Classical ML algorithms, preprocessing, metrics
- **TensorFlow/Keras**: Deep learning (LSTM networks)

### Data Processing
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations

### Visualization
- **matplotlib**: Plotting and visualization

### Utilities
- **joblib**: Model serialization
- **Jupyter Notebook**: Interactive development

## 📚 Key Concepts Demonstrated

### Machine Learning Techniques
- **Supervised Learning**: Classification and Regression
- **Ensemble Methods**: Random Forest, Gradient Boosting
- **Deep Learning**: LSTM Neural Networks
- **Time Series Analysis**: Sequence prediction

### Data Preprocessing
- Missing value imputation
- Outlier detection and removal
- Feature scaling (MinMax, Standard)
- Label encoding for categorical variables
- Data concatenation and merging

### Model Evaluation
- Train/Test splitting
- Stratified sampling
- Cross-validation ready architecture
- Multiple metrics (MAE, RMSE, Accuracy, Precision, Recall, F1)

### Best Practices
- Model persistence (saving/loading)
- Hyperparameter configuration
- Reproducibility (random_state)
- Visualization of results

## 🎓 Learning Outcomes

By exploring this repository, you will learn:

1. **Data Preprocessing**: How to clean, transform, and prepare real-world datasets
2. **Feature Engineering**: Creating meaningful features for ML models
3. **Model Comparison**: Evaluating multiple algorithms to find the best performer
4. **Time Series Prediction**: Using LSTM networks for sequential data
5. **Model Deployment**: Saving and loading trained models
6. **Performance Evaluation**: Using appropriate metrics for different problem types

## 🔮 Future Enhancements

- [ ] Add more cryptocurrency assets (Cardano, Solana, etc.)
- [ ] Implement hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- [ ] Add cross-validation for all models
- [ ] Create web interfaces using Streamlit or Flask
- [ ] Implement real-time prediction APIs
- [ ] Add more deep learning models (GRU, Transformer)
- [ ] Include feature importance analysis
- [ ] Add model interpretability (SHAP, LIME)
- [ ] Create automated ML pipeline
- [ ] Add unit tests and CI/CD

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs or issues
- Suggest new features or improvements
- Submit pull requests
- Improve documentation
- Add new ML projects

### How to Contribute

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some feature'`)
5. Push to the branch (`git push origin feature/your-feature`)
6. Open a Pull Request

## 📝 License

This repository is available for educational and research purposes. Feel free to use the code for learning and non-commercial projects.

## 👨‍💻 Author

**Hruday P**
- GitHub: [@Hruday-p](https://github.com/Hruday-p)

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Adult Census Income dataset
- Cryptocurrency exchanges for historical price data
- TensorFlow and scikit-learn communities
- Open-source contributors

## 📧 Contact

For questions, suggestions, or collaborations, feel free to open an issue or reach out through GitHub.

---

**Note**: This repository is maintained for educational purposes. The models and predictions should not be used for actual trading or financial decisions. Always perform thorough analysis and consult professionals before making financial decisions.

## 🌟 Star this Repository

If you find this repository helpful, please consider giving it a ⭐ star to show your support!
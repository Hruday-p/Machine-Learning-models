# Blockchain Price Prediction using Random Forest 📊

A machine learning project that predicts cryptocurrency prices for Ethereum, Bitcoin, and TRON using historical price data and Random Forest Regression algorithm.

## 📋 Overview

This project loads and processes historical price data from multiple CSV files for three major cryptocurrencies (Ethereum, Bitcoin, and TRON), trains Random Forest regression models on each dataset, and evaluates their performance using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) metrics.

## ✨ Features

- **Multi-Asset Support**: Processes Ethereum, Bitcoin, and TRON price data
- **Data Concatenation**: Merges multiple historical data files for each cryptocurrency
- **Advanced Data Cleaning**: Handles K/M notation and percentage symbols
- **Feature Scaling**: MinMaxScaler normalization for optimal model performance
- **Random Forest Regression**: 100-estimator ensemble model for price prediction
- **Performance Metrics**: MAE and RMSE evaluation for each asset
- **Results Export**: Saves prediction results to CSV file

## 🔧 Requirements

### Python Libraries

```bash
pandas
numpy
scikit-learn
```

### Installation

Install all dependencies using pip:

```bash
pip install pandas numpy scikit-learn
```

## 📊 Dataset

### Required Files

The project expects 9 CSV files (3 for each cryptocurrency):

#### Ethereum:
- `Ethereum Historical Data.csv`
- `Ethereum Historical Data (1).csv`
- `Ethereum Historical Data (2).csv`

#### Bitcoin:
- `Bitcoin Historical Data.csv`
- `Bitcoin Historical Data (1).csv`
- `Bitcoin Historical Data (2).csv`

#### TRON:
- `TRON Inc Stock Price History.csv`
- `TRON Inc Stock Price History (1).csv`
- `TRON Inc Stock Price History (2).csv`

### Expected Data Format

Each CSV should contain the following columns:
- **Date**: Date in MM/DD/YYYY format
- **Price**: Closing price (can include K/M notation)
- **Open**: Opening price
- **High**: Highest price
- **Low**: Lowest price
- **Vol.**: Trading volume (can include K/M notation)
- **Change %**: Percentage change (with % symbol)

## 🚀 Quick Start

### 1. Prepare Your Data

Place all 9 CSV files in the `/content/` directory (or update file paths in the code).

### 2. Run the Notebook

```bash
jupyter notebook Untitled2.ipynb
```

Or run in Google Colab directly.

### 3. View Results

The script will output a comparison table and save results to `blockchain_latency_prediction_rf_results.csv`.

## 🔄 How It Works

### 1. Data Loading and Concatenation

The `load_concat()` function:
- Loads multiple CSV files for each cryptocurrency
- Concatenates them into a single DataFrame
- Handles data cleaning and type conversions

### 2. Data Preprocessing

#### Numeric Conversion
- Converts 'K' notation to ×1000 (e.g., "50K" → 50000)
- Converts 'M' notation to ×1000000 (e.g., "2.5M" → 2500000)
- Removes commas from numbers
- Handles percentage symbols in 'Change %' column

#### Date Processing
- Converts date strings to datetime objects
- Sorts data chronologically

### 3. Feature Scaling

- Applied MinMaxScaler to normalize features to [0, 1] range
- Separate scalers for each cryptocurrency dataset
- Scaled features: `['Open', 'High', 'Low', 'Vol.', 'Change %']`

### 4. Model Training

The `train_rf()` function:
- Splits data: 80% training, 20% testing (chronological split)
- Trains Random Forest Regressor with 100 estimators
- Makes predictions on test set
- Calculates MAE and RMSE metrics

### 5. Results Compilation

- Trains models for all three cryptocurrencies
- Compiles results into a DataFrame
- Exports to CSV file

## 📈 Model Details

### Algorithm
- **Model**: Random Forest Regressor
- **Estimators**: 100 trees
- **Random State**: 42 (for reproducibility)
- **Problem Type**: Regression (continuous price prediction)

### Input Features (5)
1. **Open**: Opening price
2. **High**: Highest price in the period
3. **Low**: Lowest price in the period
4. **Vol.**: Trading volume
5. **Change %**: Percentage change

### Target Variable
- **Price**: Closing price

### Train/Test Split
- **Training**: 80% (chronologically earlier data)
- **Testing**: 20% (chronologically later data)

## 📊 Performance Metrics

### Mean Absolute Error (MAE)
Measures the average absolute difference between predicted and actual prices.

### Root Mean Squared Error (RMSE)
Measures the square root of average squared differences, penalizing larger errors more.

### Example Results

```
     Asset      RF MAE      RF RMSE
0  Ethereum   126.908700  156.875963
1  Bitcoin    829.473091  994.466779
2  TRON         0.748435    0.979000
```

**Interpretation:**
- **Ethereum**: Average prediction error of ~$127
- **Bitcoin**: Average prediction error of ~$829
- **TRON**: Average prediction error of ~$0.75

## 🎯 Use Cases

- **Price Forecasting**: Predict future cryptocurrency prices
- **Trading Strategies**: Inform buy/sell decisions with predictions
- **Risk Assessment**: Evaluate volatility using prediction errors
- **Market Analysis**: Compare prediction accuracy across different assets
- **Research**: Study cryptocurrency price patterns and trends

## 🔄 Customization

### Adjust Train/Test Split

```python
split = int(len(df) * 0.7)  # Change to 70/30 split
```

### Modify Random Forest Parameters

```python
model = RandomForestRegressor(
    n_estimators=200,      # Increase trees
    max_depth=10,          # Limit tree depth
    min_samples_split=5,   # Minimum samples to split
    random_state=42
)
```

### Add More Features

```python
features = ['Open', 'High', 'Low', 'Vol.', 'Change %', 'MovingAvg', 'RSI']
```

### Change File Paths

Update the file path lists at the beginning:
```python
e_files = ["path/to/eth1.csv", "path/to/eth2.csv", "path/to/eth3.csv"]
```

## 📂 Output Files

### blockchain_latency_prediction_rf_results.csv

Contains comparison of model performance across all three cryptocurrencies:
- Asset name
- Random Forest MAE
- Random Forest RMSE

## 🛠️ Troubleshooting

### File Not Found Error
- Verify all 9 CSV files are in the correct directory
- Check file names match exactly (case-sensitive)
- Update file paths if using different directory structure

### Data Parsing Errors
- Ensure date format is MM/DD/YYYY
- Check that numeric columns use K/M notation correctly
- Verify percentage values include % symbol

### Low Model Performance
- Try increasing `n_estimators` (more trees)
- Add more features (technical indicators, moving averages)
- Increase training data size
- Tune hyperparameters using GridSearchCV

### Memory Issues
- Reduce number of estimators
- Process one cryptocurrency at a time
- Use smaller time windows for data

## 📊 Data Preprocessing Pipeline

```
Raw CSV Files
    ↓
Load & Concatenate
    ↓
Clean Numeric Values (K/M notation)
    ↓
Convert Percentages
    ↓
Parse Dates
    ↓
Sort Chronologically
    ↓
Scale Features (MinMaxScaler)
    ↓
Train/Test Split (80/20)
    ↓
Random Forest Model
    ↓
Predictions & Evaluation
```

## 🔐 Best Practices

### Data Quality
- Ensure no missing values in critical columns
- Verify date continuity (no large gaps)
- Check for outliers that might affect predictions

### Model Validation
- Use walk-forward validation for time series
- Consider cross-validation with time series splits
- Compare against baseline models (e.g., naive forecast)

### Feature Engineering
- Add technical indicators (RSI, MACD, Bollinger Bands)
- Include lagged features (previous day prices)
- Consider external factors (market sentiment, news)

## 📚 Technical Details

- **Framework**: scikit-learn
- **Language**: Python 3.x
- **Environment**: Jupyter Notebook / Google Colab
- **Problem Type**: Time Series Regression
- **Data Structure**: Chronologically ordered price data

## 🚀 Future Enhancements

- [ ] Add LSTM/GRU neural networks for comparison
- [ ] Implement technical indicators as features
- [ ] Add visualization of predictions vs actual prices
- [ ] Include confidence intervals for predictions
- [ ] Implement hyperparameter tuning
- [ ] Add real-time prediction capabilities
- [ ] Support for more cryptocurrencies

## 👨‍💻 Author

Developed for cryptocurrency price prediction research and educational purposes using Random Forest regression.

## 📄 License

This project is available for educational and research purposes.

## 🙏 Acknowledgments

- Historical price data from cryptocurrency exchanges
- scikit-learn library for machine learning tools
- pandas for data manipulation

---

**Note**: Cryptocurrency prices are highly volatile and influenced by numerous external factors. This model is for educational purposes and should not be used as financial advice. Always perform thorough analysis before making investment decisions.
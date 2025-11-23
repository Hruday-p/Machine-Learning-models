# Temperature Forecasting using LSTM

A deep learning project that uses Long Short-Term Memory (LSTM) neural networks to forecast temperature based on historical hourly temperature data.

## 📋 Overview

This project demonstrates time series forecasting by predicting the next hour's temperature using the past 24 hourly temperature readings. The model employs a stacked LSTM architecture to capture temporal patterns in temperature data.

## ✨ Features

- **Time Series Preprocessing**: Automated data cleaning and timestamp conversion
- **Data Scaling**: MinMaxScaler normalization for optimal neural network performance
- **Sequence Generation**: Sliding window approach to create supervised learning sequences
- **Stacked LSTM Architecture**: Two-layer LSTM network with 50 units each
- **Model Evaluation**: RMSE (Root Mean Squared Error) metric for performance assessment
- **Visualization**: Interactive plots comparing actual vs predicted temperatures

## 🔧 Requirements

This project requires the following Python libraries:

```bash
pandas
numpy
datetime
tensorflow
scikit-learn
matplotlib
```

### Installation

Install all dependencies using pip:

```bash
pip install pandas numpy tensorflow scikit-learn matplotlib
```

## 📊 Dataset

The code expects a CSV file containing:
- **Timestamp**: Date and time of temperature reading
- **Temperature**: Temperature value in your preferred unit

Dataset path in code: `content/predictive-maintenance-dataset.csv`

## 🚀 Usage

1. **Prepare your data**: Ensure your CSV file has `Timestamp` and `Temperature` columns
2. **Update dataset path**: Modify the file path in the code if needed
3. **Run the script**: Execute the Python file

```bash
python temperature_forecast_using_lstm.py
```

4. **View results**: Check the console for RMSE values and the generated plot

## 🧠 Model Architecture

```
Input Layer (24 timesteps, 1 feature)
    ↓
LSTM Layer (50 units, return_sequences=True)
    ↓
LSTM Layer (50 units)
    ↓
Dense Layer (1 unit)
```

### Hyperparameters

- **Timesteps**: 24 (uses 24 previous hours to predict next hour)
- **LSTM Units**: 50 per layer
- **Epochs**: 20
- **Batch Size**: 32
- **Validation Split**: 10% of training data
- **Train/Test Split**: 80/20

## 📈 How It Works

1. **Data Loading**: Load temperature data from CSV file
2. **Preprocessing**: Convert timestamps to datetime format
3. **Train-Test Split**: Split data into 80% training and 20% testing
4. **Scaling**: Normalize temperature values to [0, 1] range
5. **Sequence Creation**: Generate sequences of 24 timesteps for input
6. **Model Training**: Train stacked LSTM model on prepared sequences
7. **Prediction**: Generate predictions on test data
8. **Evaluation**: Calculate RMSE and visualize results

## 📊 Output

The script generates:
- **RMSE Score**: Printed to console showing prediction accuracy
- **Visualization Plot**: Comparison of actual vs predicted temperatures (last 500 test points)

## 🎯 Model Performance

The model's performance is evaluated using RMSE (Root Mean Squared Error), which measures the average deviation between predicted and actual temperatures. Lower RMSE indicates better prediction accuracy.

## 🔄 Customization

You can customize the following parameters:

- **Timesteps**: Change `timesteps = 24` to use different lookback periods
- **LSTM Units**: Modify the number of units in LSTM layers
- **Epochs**: Adjust training duration by changing `epochs=20`
- **Train/Test Ratio**: Modify `trainsize = int(len(data) * 0.8)`

## 📝 Example Output

```
Test RMSE: X.XX
```

A matplotlib plot window will display showing:
- Blue line: Actual temperatures
- Red line: Predicted temperatures

## 🛠️ Troubleshooting

- **File not found error**: Verify the dataset path is correct
- **Memory issues**: Reduce batch size or timesteps
- **Poor predictions**: Try increasing epochs or adjusting LSTM units

## 📚 Technical Details

- **Framework**: TensorFlow/Keras
- **Neural Network Type**: Recurrent Neural Network (LSTM)
- **Problem Type**: Time Series Forecasting
- **Learning Type**: Supervised Learning

## 👨‍💻 Author

Developed for educational purposes to demonstrate temperature forecasting using LSTM neural networks.

## 📄 License

This project is available for educational and research purposes.

---

**Note**: Make sure to adjust the dataset path (`content/predictive-maintenance-dataset.csv`) according to your file location before running the script.
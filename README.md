# ML Project - Classification & Regression with Streamlit

A comprehensive machine learning project that provides:
- **Exploratory Data Analysis (EDA)** using ydata-profiling
- **Classification Models** (Random Forest, Logistic Regression)
- **Regression Models** (Random Forest Regressor, Linear Regression)
- **Interactive Web Interface** built with Streamlit

## Features

1. **Data Overview**: Quick statistics, missing values, and dataset preview
2. **EDA**: Full ydata-profiling report with comprehensive data analysis
3. **Classification**: Train and evaluate classification models with metrics and visualizations
4. **Regression**: Train and evaluate regression models with performance metrics

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Make sure your dataset (`Telco_Cusomer_Churn.csv`) is in the project directory
2. Run the Streamlit app:
```bash
streamlit run main.py
```

3. The app will open in your browser automatically

## Dataset

The project uses the Telco Customer Churn dataset with:
- 7043 rows
- 21 columns
- Mix of categorical and numerical features
- Target variable: Churn (for classification)
- Numerical targets: MonthlyCharges, TotalCharges (for regression)

## Project Structure

```
ml project/
├── main.py                    # Main Streamlit application
├── requirements.txt           # Python dependencies
├── Telco_Cusomer_Churn.csv    # Dataset
└── README.md                  # This file
```

## Features of the App

### Data Overview
- Dataset statistics
- Column information
- Missing values analysis
- Data preview

### Exploratory Data Analysis
- Full ydata-profiling report
- Interactive visualizations
- Correlation matrices
- Distribution plots

### Classification
- Multiple model options (Random Forest, Logistic Regression)
- Feature selection
- Model evaluation metrics
- Confusion matrix visualization
- Feature importance plots

### Regression
- Multiple model options (Random Forest, Linear Regression)
- Feature selection
- Model evaluation metrics (R², RMSE, MAE, MSE)
- Prediction vs Actual plots
- Residuals analysis
- Feature importance plots

## Requirements

- Python 3.8+
- Streamlit
- pandas
- numpy
- ydata-profiling
- plotly
- scikit-learn


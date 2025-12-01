# prediction-tesla
Predicting Tesla stock prices trends using logistic regression and decision trees.

# Stock Price Prediction – Tesla

This repository contains a project for predicting Tesla stock price trends using two machine learning models: logistic regression and decision trees. The goal is to compare their performance in predicting whether the stock price will rise or fall. 

## Objectives 

- Evaluate the ability of classification models to predict Tesla’s future price direction.
- Compare models using metrics: accuracy, precision, recall.
- Identify model limitations and potential improvements.

## Script Overview

- `stock_tesla.py` – Displays and saves metrics and graphs of the prediction models used.

## Data

- 5 years of historical data for TSLA via yfinance.
- Technical indicators: SMA (moving averages), RSI, volumes, and price derivatives.
- Target variable : price direction for the next period (up/down).

## Models

- Logistic Regression – linear model.
- Decision Trees – non-linear model capturing complex interactions.
- Hyperparameters optimized with cross-validation to limit overfitting.

## Results

- Decision trees slightly outperform logistic regression in overall accuracy.
- Logistic regression provides better balance between classes.
- Stock trend prediction is inherently difficult, sometimes near random.

## Dependencies

- Python >= 3.8 
- pandas  
- numpy
- scikit-learn
- yfinance
- matplotlib
- seaborn 

You can install the required packages with :

```bash
pip install pandas numpy scikit-learn yfinance matplotlib seaborn
```

## Usage

```bash
stock_tesla.py
```

Download TSLA data using yfinance.

Run the script:

python script/stock_tsla.py

Results (metrics, confusion matrices, charts) will be saved in the results/ folder.

## Notes

- Data files are not included due to size. They can be regenerated using yfinance.

## Author

[Yasmine Aissa / Yasmine56]

# Quant Research Assignment – Volatility Pairs Trading

## Overview

This repository contains two implementations of a volatility pairs trading strategy using minute-level implied volatility (IV) data for the Nifty and Bank Nifty indices. The goal is to test the hypothesis that the volatilities of these indices, due to their constituent overlap, will tend to move together, and that any dispersion between them can be traded profitably.

## Data

- **data.parquet**: Contains minute-level IVs for Nifty and Bank Nifty, and Time To Expiry (TTE) for each row.
- Indian market trading hours: 09:15 to 15:30 (weekdays)

## Strategy 1: Z-Score Based Model (`z_score_strategy.py`)

This is the base model. It uses the rolling z-score of the spread between Bank Nifty IV and Nifty IV to generate trading signals.

- **Spread Calculation:**  
    Spread = Bank Nifty IV - Nifty IV
- **Z-Score Calculation:**  
  The spread is standardized using a rolling mean and standard deviation.
- **Signal Generation:**  
  - **Enter Long:** z-score < -2
  - **Enter Short:** z-score > 2
  - **Exit:** |z-score| < 0.5
- **PnL Calculation:**  
    P/L = Spread*(Time To Expiry)^0.7
- **Missing Data Handling:**  
  Short gaps are forward-filled; longer gaps are imputed using a GARCH(1,1) model or linear interpolation as fallback.
- **Performance Metrics:**  
  Total PnL, number of trades, Sharpe ratio, drawdown, win rate.
- **Visualization:**  
  Plots spread, z-score, positions, cumulative PnL, and drawdown.

## Strategy 2: Kalman Filter Model (`kalman_strategy.py`)

This is the improved model. It uses a Kalman filter to dynamically estimate the relationship between Nifty and Bank Nifty IVs, and generates signals based on the normalized innovation (model-based z-score).

- **Kalman Filter:**  
  Estimates a time-varying hedge ratio (beta) and intercept (alpha) between the two IV series.
- **Innovation:**  
  The innovation (residual) is the difference between the observed and predicted Bank Nifty IV.
- **Normalized Innovation:**  
  The innovation is divided by its rolling standard deviation (estimated by the Kalman filter), yielding a model-based z-score.
- **Signal Generation:**  
  - **Enter Long:** normalized innovation < -1.5
  - **Enter Short:** normalized innovation > 1.5
  - **Exit:** |normalized innovation| < 0.5
- **PnL Calculation:**  
  Same as the base model.
- **Missing Data Handling:**  
  Block-wise Kalman imputation, with linear interpolation and edge-case handling.
- **Performance Metrics:**  
  Same as the base model.
- **Visualization:**  
  Plots innovation, normalized innovation, positions, cumulative PnL, drawdown, and Kalman filter parameters.

## How to Run

1. Place `data.parquet` in the same directory as the scripts.
2. Run each script:
   ```
   python z_score_strategy.py
   python kalman_strategy.py
   ```
3. Each script will print performance metrics and save plots to their respective output directories.

## Assumptions and Notes

- All calculations are restricted to Indian market trading hours.
- Missing data is handled robustly using forward fill, GARCH, Kalman, and interpolation as appropriate.
- Trade costs are included in PnL calculations.
- The Kalman filter model is expected to be more adaptive and robust to changing market conditions than the rolling z-score model.

## Results

Both scripts print a summary of performance metrics (P/L, Sharpe, Drawdown, etc.) after execution. Compare these metrics to evaluate which model performs better on the provided data.

## TL;DR: Results Comparison

| Model                | Total P/L | Number of Trades | Sharpe Ratio | Max Drawdown (abs) | Max Drawdown (%) | Win Rate |
|----------------------|-----------|------------------|--------------|--------------------|------------------|----------|
| Z-Score (Base)       | 31.26     | 2,763            | -0.25        | -8.02              | -0.51            | 0.26     |
| Kalman (Improved)    | 79.11     | 5,374            | 1.73         | -10.90             | -1.07            | 0.25     |

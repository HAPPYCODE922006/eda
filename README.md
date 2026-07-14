# Air Quality Index (AQI) Forecasting

## Project Overview
Exploratory Data Analysis (EDA) of air quality data across Indian cities (2015–2020) 
to understand pollution patterns and build a predictive model for AQI forecasting.

## Dataset
- **Source:** [Kaggle / CPCB India]
- **Period:** January 2015 – July 2020
- **Records:** 24,850 rows after cleaning
- **Features:** PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene
- **Target:** AQI (Air Quality Index)

## EDA Findings
1. **PM10 is the strongest AQI predictor** (correlation = 0.80)
2. **PM2.5 and CO are secondary drivers** (correlation = 0.66, 0.68)
3. **AQI is right-skewed** with frequent extreme pollution spikes
4. **68% of days are Moderate/Satisfactory**, but 25% are Poor or worse
5. **Strong seasonal pattern:** winter peaks, summer troughs
6. **2020 shows lowest AQI** — likely COVID-19 lockdown effect

## Plots
| Plot | Description |
|------|-------------|
| Correlation Heatmap | Pollutant correlations with AQI |
| AQI Distribution | Histogram with KDE |
| AQI Categories | Count by health category |
| Pollutant Boxplot | Outlier detection (log scale) |
| PM2.5 vs AQI | Scatter with category colors |
| AQI Trend | Monthly average over time |

## Installation
```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib

# 🌫️ AQI Predictor

A machine learning project that predicts India's Air Quality Index (AQI) from 11 pollutant
concentration readings, with an interactive Streamlit dashboard for exploring predictions.

## Project Overview

Air quality monitoring stations record concentrations of individual pollutants (PM2.5, PM10,
NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene), but the composite Air Quality Index (AQI)
is what actually communicates health risk to the public. This project trains a Random Forest
Regressor to learn the mapping from raw pollutant readings to AQI, then wraps it in a Streamlit
app that lets a user enter (or preset) pollutant levels and instantly see the predicted AQI,
its health category, and guidance.

**Pipeline:**
1. `clean_data.py` — cleans the raw Kaggle dataset into a model-ready CSV.
2. `notebooks/eda.ipynb` — exploratory data analysis: distributions, missingness, correlations,
   city/time trends.
3. `model.py` — trains a `RandomForestRegressor` and saves it as `aqi_model.pkl`.
4. `app.py` — Streamlit dashboard that loads `aqi_model.pkl` and serves live predictions.

## Dataset

**Source:** [Air Quality Data in India (2015–2020)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
on Kaggle (`city_day.csv`), originally published by the Central Pollution Control Board (CPCB),
Government of India.

The dataset contains daily average pollutant readings and AQI for 26 Indian cities
(e.g. Delhi, Mumbai, Bengaluru, Chennai, Kolkata) between 2015 and 2020.

**Columns used for modeling:**

| Column | Description | Unit |
|---|---|---|
| PM2.5 | Fine particulate matter | µg/m³ |
| PM10 | Coarse particulate matter | µg/m³ |
| NO | Nitric oxide | ppb |
| NO2 | Nitrogen dioxide | ppb |
| NOx | Nitrogen oxides | ppb |
| NH3 | Ammonia | ppb |
| CO | Carbon monoxide | mg/m³ |
| SO2 | Sulphur dioxide | ppb |
| O3 | Ozone | ppb |
| Benzene | Benzene | µg/m³ |
| Toluene | Toluene | µg/m³ |
| AQI | Target — Air Quality Index | — |

**Getting the data:** download `city_day.csv` from the Kaggle link above and place it at
`data/city_day.csv` in this project (not included in this submission due to size/licensing —
see [Steps to Run](#steps-to-run) below).

## Installation Requirements

- Python 3.9+
- Dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Steps to Run

1. **Get the data.** Download `city_day.csv` from
   [Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) and save it to
   `data/city_day.csv`.

2. **Clean the data.**
   ```bash
   python clean_data.py
   ```
   This produces `data/cleaned_aqi.csv`, filling missing pollutant readings with per-city
   medians and dropping rows with no AQI label.

3. *(Optional)* **Explore the data.**
   ```bash
   jupyter notebook notebooks/eda.ipynb
   ```

4. **Train the model.**
   ```bash
   python train_model.py
   ```
   This trains a `RandomForestRegressor` on the cleaned data, prints RMSE / MAE / R², and saves
   the fitted model to `aqi_model.pkl` in the project root.

5. **Run the dashboard.**
   ```bash
   streamlit run app.py
   ```
   Open the local URL Streamlit prints (usually `http://localhost:8501`). Use the sidebar
   sliders or a quick preset ("Good day", "Moderate day", "Polluted day", "Hazardous day") and
   click **Predict AQI**.

## App Features

- 🔮 **Predict tab** — sliders for all 11 pollutants, gauge visualization, predicted AQI with
  health category and advice, downloadable CSV of the result, feature-importance chart.
- 📈 **History & Trend tab** — line chart and table of every prediction made in the session,
  downloadable as CSV.
- ℹ️ **Model Info tab** — model type, feature list, and feature importances.

## Project Structure

```
.
├── app.py                 # Streamlit dashboard
├── clean_data.py           # Raw -> cleaned dataset
├── model.py          # Trains & saves the Random Forest model
├── requirements.txt
├── README.md
├── notebooks/
│   └── eda.ipynb            # Exploratory data analysis
└── data/
    ├── city_day.csv          # Raw Kaggle dataset (download separately)
    └── cleaned_aqi.csv       # Produced by clean_data.py
```

## Model

- **Algorithm:** Random Forest Regressor (scikit-learn), 100 trees, `random_state=42`
- **Features:** the 11 pollutant columns listed above
- **Target:** AQI
- **Evaluation:** 80/20 train/test split; RMSE, MAE, and R² printed at training time



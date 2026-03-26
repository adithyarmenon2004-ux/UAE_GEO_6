# 🇦🇪 UAE Customer Intelligence Dashboard

A beautiful ML-powered Streamlit dashboard for marketing customer analytics.

## Features
- 📋 **Overview** — dataset stats, distributions, buy-intent split
- 🎯 **Classification** — Random Forest purchase prediction with ROC curve & feature importance
- 👥 **Segmentation** — KMeans clustering with PCA visualization & segment profiles
- 💰 **Spend Forecast** — Random Forest regression with actual vs predicted plots
- 🔮 **Predict New** — upload new customers and download scored predictions

## Required CSV Columns
| Column | Type | Description |
|---|---|---|
| `Will_Buy` | 0 / 1 | Classification target |
| `Max_Spend` | numeric | Regression target |
| other columns | any | Features for the model |

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud
1. Push this folder to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → select `app.py`
4. Click **Deploy**

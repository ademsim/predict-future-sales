from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import streamlit as st

st.set_page_config(page_title="Future Sales Predictor")

BASE = Path(__file__).parent


@st.cache_resource
def load_model():
    return pickle.load(open(BASE / "futuresales_model.pkl", "rb"))


model = load_model()

st.title("Future Sales Predictor")
st.write(
    "A gradient boosting model (trained on the Kaggle Predict Future Sales dataset, 1C Company) predicts how many "
    "units of an item a shop will sell next month, mainly from how many units it sold the previous month."
)

col1, col2 = st.columns(2)
with col1:
    shop_id = st.number_input("Shop ID (0-59)", 0, 59, 5)
    item_id = st.number_input("Item ID (0-22169)", 0, 22169, 5037)
with col2:
    prev_month_cnt = st.slider("Units sold last month, this shop-item pair", 0, 20, 1)
    month = st.selectbox("Target month", list(range(1, 13)), index=10)

year = 2015
date_block_num = (year - 2013) * 12 + (month - 1)

if st.button("Predict"):
    row = pd.DataFrame([{
        "date_block_num": date_block_num,
        "shop_id": shop_id,
        "item_id": item_id,
        "prev_month_cnt": prev_month_cnt,
        "month": month,
        "year": year,
    }])[list(model.feature_names_in_)]

    pred = float(np.clip(model.predict(row)[0], 0, 20))
    st.success(f"Predicted units sold next month: **{pred:.1f}**")

    curve = pd.DataFrame([{**row.iloc[0].to_dict(), "prev_month_cnt": p} for p in range(0, 21)])[list(model.feature_names_in_)]
    curve["prediction"] = np.clip(model.predict(curve), 0, 20)
    st.line_chart(curve.set_index("prev_month_cnt")["prediction"])

st.caption(
    "Model: HistGradientBoosting with a single lag feature (previous month's sales for the same shop-item pair). "
    "Predictions are clipped to [0, 20], matching the competition's scoring rule. The chart shows how the "
    "prediction changes if last month's sales had been different, everything else held fixed."
)

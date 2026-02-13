import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Sneaker Resale Predictor",
    page_icon="👟",
    layout="centered"
)

model = joblib.load("model.pkl")

# ===== HEADER =====
st.markdown("""
<style>
.big-title {
    font-size:40px;
    font-weight:700;
}
.card {
    background-color:#111;
    padding:20px;
    border-radius:15px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">👟 Sneaker Resale Price Predictor</div>', unsafe_allow_html=True)
st.caption("ระบบทำนายราคาขายต่อรองเท้า Sneaker ด้วย Machine Learning")

# ===== MODEL INFO =====
col1, col2 = st.columns(2)
col1.metric("Model", "Multiple Linear Regression")
col2.metric("R² Score", "≈ 0.87")

st.divider()

# ===== INPUT =====
st.subheader("📝 กรอกข้อมูลรองเท้า")

retail_price = st.number_input("ราคาตอนเปิดตัว (บาท)", 1000, 20000, 4000)
production_qty = st.number_input("จำนวนผลิต", 10000, 500000, 200000)
release_year = st.selectbox("ปีที่ออก", list(range(2018, 2026)))
brand_popularity = st.slider("ความนิยมแบรนด์ (1–10)", 1, 10, 6)
condition = st.slider("สภาพสินค้า (1–5)", 1, 5, 5)

# ===== PREDICTION =====
if st.button("🔮 ทำนายราคา"):
    input_df = pd.DataFrame([{
        "retail_price": retail_price,
        "production_qty": production_qty,
        "release_year": release_year,
        "brand_popularity": brand_popularity,
        "condition": condition
    }])

    prediction = model.predict(input_df)[0]
    profit = prediction - retail_price

    st.success(f"💰 ราคาขายต่อโดยประมาณ: {prediction:,.0f} บาท")

    if profit > 0:
        st.info(f"📈 กำไรโดยประมาณ: {profit:,.0f} บาท")
    else:
        st.warning(f"📉 ขาดทุนโดยประมาณ: {abs(profit):,.0f} บาท")
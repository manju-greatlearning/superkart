import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="SuperKart Sales Forecast", layout="centered")
st.title("SuperKart Sales Forecast")
st.write("Predict product-store sales revenue using the registered SuperKart model.")

MODEL_REPO_ID = st.secrets.get("MODEL_REPO_ID", "YOUR_HF_USERNAME/superkart-sales-model")
MODEL_FILENAME = "superkart_sales_model.joblib"

@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
    return joblib.load(model_path)

model = load_model()

product_weight = st.number_input("Product Weight", min_value=0.0, value=12.5)
product_sugar_content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
product_allocated_area = st.number_input("Product Allocated Area", min_value=0.0, max_value=1.0, value=0.05)
product_type = st.selectbox("Product Type", ["Frozen Foods", "Dairy", "Canned", "Baking Goods", "Health and Hygiene", "Snack Foods", "Meat", "Household", "Hard Drinks", "Fruits and Vegetables", "Breads", "Soft Drinks", "Breakfast", "Seafood", "Starchy Foods", "Others"])
product_mrp = st.number_input("Product MRP", min_value=0.0, value=150.0)
store_id = st.selectbox("Store ID", ["OUT001", "OUT002", "OUT003", "OUT004"])
store_establishment_year = st.selectbox("Store Establishment Year", [1987, 1998, 1999, 2009])
store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
store_location_city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
store_type = st.selectbox("Store Type", ["Departmental Store", "Supermarket Type1", "Supermarket Type2", "Food Mart"])

if st.button("Predict Sales"):
    row = pd.DataFrame([{
        "Product_Weight": product_weight,
        "Product_Sugar_Content": product_sugar_content,
        "Product_Allocated_Area": product_allocated_area,
        "Product_Type": product_type,
        "Product_MRP": product_mrp,
        "Store_Id": store_id,
        "Store_Establishment_Year": store_establishment_year,
        "Store_Size": store_size,
        "Store_Location_City_Type": store_location_city_type,
        "Store_Type": store_type,
    }])
    pred = model.predict(row)[0]
    st.success(f"Predicted Product Store Sales Total: {pred:,.2f}")

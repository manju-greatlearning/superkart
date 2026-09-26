import joblib
import pandas as pd
import gradio as gr
from huggingface_hub import hf_hub_download

MODEL_REPO_ID = "manjunathans/superkart-sales-model"
MODEL_FILENAME = "superkart_sales_model.joblib"

model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
model = joblib.load(model_path)


def predict_sales(
    product_weight,
    product_sugar_content,
    product_allocated_area,
    product_type,
    product_mrp,
    store_id,
    store_establishment_year,
    store_size,
    store_location_city_type,
    store_type,
):
    row = pd.DataFrame([
        {
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
        }
    ])
    prediction = model.predict(row)[0]
    return f"Predicted Product Store Sales Total: {prediction:,.2f}"


demo = gr.Interface(
    fn=predict_sales,
    inputs=[
        gr.Number(label="Product Weight", value=12.5),
        gr.Dropdown(["Low Sugar", "Regular", "No Sugar"], label="Product Sugar Content", value="Low Sugar"),
        gr.Number(label="Product Allocated Area", value=0.05),
        gr.Dropdown([
            "Frozen Foods", "Dairy", "Canned", "Baking Goods", "Health and Hygiene",
            "Snack Foods", "Meat", "Household", "Hard Drinks", "Fruits and Vegetables",
            "Breads", "Soft Drinks", "Breakfast", "Seafood", "Starchy Foods", "Others"
        ], label="Product Type", value="Snack Foods"),
        gr.Number(label="Product MRP", value=150.0),
        gr.Dropdown(["OUT001", "OUT002", "OUT003", "OUT004"], label="Store ID", value="OUT004"),
        gr.Dropdown([1987, 1998, 1999, 2009], label="Store Establishment Year", value=2009),
        gr.Dropdown(["Small", "Medium", "High"], label="Store Size", value="Medium"),
        gr.Dropdown(["Tier 1", "Tier 2", "Tier 3"], label="Store Location City Type", value="Tier 2"),
        gr.Dropdown(["Departmental Store", "Supermarket Type1", "Supermarket Type2", "Food Mart"], label="Store Type", value="Supermarket Type2"),
    ],
    outputs=gr.Textbox(label="Sales Forecast"),
    title="SuperKart Sales Forecasting App",
    description="Enter product and store attributes to forecast total product-store sales revenue.",
)

if __name__ == "__main__":
    demo.launch()

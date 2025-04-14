# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib

# # Load the scaler and model
# scaler = joblib.load('scaler.pkl')
# model = joblib.load('model.pkl')

# # App title
# st.title("Retail Price Prediction App")

# # Sidebar for user input
# st.sidebar.header("Input Parameters")

# def user_input_features():
#     product_category_name = st.sidebar.selectbox(
#         "Product Category",
#         [
#             "bed_bath_table", "garden_tools", "consoles_games", "health_beauty",
#             "cool_stuff", "perfumery", "computers_accessories", "watches_gifts", "furniture_decor"
#         ]
#     )
#     total_price = st.sidebar.number_input("Total Price", min_value=0.0, value=100.0, step=1.0)
#     freight_price = st.sidebar.number_input("Freight Price", min_value=0.0, value=10.0, step=1.0)
#     unit_price = st.sidebar.number_input("Unit Price", min_value=0.0, value=20.0, step=1.0)
#     product_name_lenght = st.sidebar.number_input("Product Name Length", min_value=0, value=40, step=1)
#     product_description_lenght = st.sidebar.number_input("Product Description Length", min_value=0, value=160, step=1)
#     product_photos_qty = st.sidebar.number_input("Product Photos Qty", min_value=0, value=1, step=1)
#     product_weight_g = st.sidebar.number_input("Product Weight (g)", min_value=0.0, value=500.0, step=1.0)
#     product_score = st.sidebar.number_input("Product Score", min_value=0.0, max_value=5.0, value=4.5, step=0.1)
#     customers = st.sidebar.number_input("Customers", min_value=0, value=10, step=1)
#     weekday = st.sidebar.number_input("Weekday", min_value=0, max_value=6, value=0, step=1)
#     weekend = st.sidebar.number_input("Weekend (0 or 1)", min_value=0, max_value=1, value=0, step=1)
#     holiday = st.sidebar.number_input("Holiday (0 or 1)", min_value=0, max_value=1, value=0, step=1)
#     month = st.sidebar.number_input("Month", min_value=1, max_value=12, value=1, step=1)
#     year = st.sidebar.number_input("Year", min_value=2000, max_value=2100, value=2025, step=1)
#     s = st.sidebar.number_input("S", min_value=0.0, value=1.0, step=0.1)
#     volume = st.sidebar.number_input("Volume", min_value=0.0, value=1.0, step=0.1)
#     comp_1 = st.sidebar.number_input("Comp 1", min_value=0.0, value=100.0, step=1.0)
#     ps1 = st.sidebar.number_input("PS1", min_value=0.0, value=4.0, step=0.1)
#     fp1 = st.sidebar.number_input("FP1", min_value=0.0, value=15.0, step=1.0)
#     comp_2 = st.sidebar.number_input("Comp 2", min_value=0.0, value=200.0, step=1.0)
#     ps2 = st.sidebar.number_input("PS2", min_value=0.0, value=4.0, step=0.1)
#     fp2 = st.sidebar.number_input("FP2", min_value=0.0, value=20.0, step=1.0)
#     comp_3 = st.sidebar.number_input("Comp 3", min_value=0.0, value=50.0, step=1.0)
#     ps3 = st.sidebar.number_input("PS3", min_value=0.0, value=4.0, step=0.1)
#     fp3 = st.sidebar.number_input("FP3", min_value=0.0, value=15.0, step=1.0)
#     lag_price = st.sidebar.number_input("Lag Price", min_value=0.0, value=40.0, step=1.0)

#     data = {
#         'product_category_name': product_category_name,
#         'total_price': total_price,
#         'freight_price': freight_price,
#         'unit_price': unit_price,
#         'product_name_lenght': product_name_lenght,
#         'product_description_lenght': product_description_lenght,
#         'product_photos_qty': product_photos_qty,
#         'product_weight_g': product_weight_g,
#         'product_score': product_score,
#         'customers': customers,
#         'weekday': weekday,
#         'weekend': weekend,
#         'holiday': holiday,
#         'month': month,
#         'year': year,
#         's': s,
#         'volume': volume,
#         'comp_1': comp_1,
#         'ps1': ps1,
#         'fp1': fp1,
#         'comp_2': comp_2,
#         'ps2': ps2,
#         'fp2': fp2,
#         'comp_3': comp_3,
#         'ps3': ps3,
#         'fp3': fp3,
#         'lag_price': lag_price
#     }
#     df = pd.DataFrame(data, index=[0])
#     df['log_price'] = np.log(df['unit_price'])
    
#     return df

# # Map product_category_name to numeric values as used in the model
# category_map = {
#     "bed_bath_table": 1,
#     "garden_tools": 2,
#     "consoles_games": 3,
#     "health_beauty": 4,
#     "cool_stuff": 5,
#     "perfumery": 6,
#     "computers_accessories": 7,
#     "watches_gifts": 8,
#     "furniture_decor": 9
# }

# # Main app
# df_input = user_input_features()
# df_input['product_category_name'] = df_input['product_category_name'].map(category_map)

# # Scale the input features
# scaled_input = scaler.transform(df_input)

# # Predict
# if st.button("Predict Quantity"):
#     prediction = model.predict(scaled_input)
#     st.success(f"Predicted Quantity: {prediction[0]:.2f}")

# # Footer
# st.markdown("### About")
# st.markdown("This app predicts the quantity of products based on various features.")




import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the scaler and model
scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')

# App title
st.title("🛒 Retail Price Prediction App")

# Product category mapping
category_map = {
    "bed_bath_table": 1,
    "garden_tools": 2,
    "consoles_games": 3,
    "health_beauty": 4,
    "cool_stuff": 5,
    "perfumery": 6,
    "computers_accessories": 7,
    "watches_gifts": 8,
    "furniture_decor": 9
}

# Input form
st.header("Enter Product Features")

def user_input_features():
    col1, col2, col3 = st.columns(3)

    with col1:
        product_category_name = st.selectbox(
            "Product Category",
            list(category_map.keys())
        )
        total_price = st.number_input("Total Price", min_value=19.9, max_value=12095.0, value=100.0, step=1.0)
        freight_price = st.number_input("Freight Price", min_value=0.0, max_value=79.76, value=10.0, step=1.0)
        unit_price = st.number_input("Unit Price", min_value=19.9, max_value=364.0, value=20.0, step=1.0)
        product_name_lenght = st.number_input("Product Name Length", min_value=29, max_value=60, value=40, step=1)
        product_description_lenght = st.number_input("Product Description Length", min_value=100, max_value=3006, value=160, step=1)
        product_photos_qty = st.number_input("Product Photos Qty", min_value=1, max_value=8, value=1, step=1)
        product_weight_g = st.number_input("Product Weight (g)", min_value=100.0, max_value=9750.0, value=500.0, step=1.0)
        product_score = st.number_input("Product Score", min_value=3.3, max_value=4.5, value=4.5, step=0.1)
    with col2:    
        
        customers = st.number_input("Customers", min_value=1, max_value=339, value=10, step=1)
        weekday = st.number_input("Weekday", min_value=20, max_value=23, value=20, step=1)
        weekend = st.number_input("Weekend", min_value=8, max_value=10, value=8, step=1)
        holiday = st.number_input("Holiday", min_value=0, max_value=4, value=0, step=1)
        month = st.number_input("Month", min_value=1, max_value=12, value=1, step=1)
        year = st.number_input("Year", min_value=2017, max_value=2018, value=2018, step=1)
        s = st.number_input("S", min_value=0.484261501, max_value=100.0, value=1.0, step=0.1)
        volume = st.number_input("Volume", min_value=640.0, max_value=32736.0, value=1000.0, step=100.0)
        comp_1 = st.number_input("Comp 1", min_value=19.9, max_value=349.9, value=100.0, step=1.0)
    with col3:    
        ps1 = st.number_input("PS1", min_value=3.7, max_value=4.5, value=4.0, step=0.1)
        fp1 = st.number_input("FP1", min_value=0.095438596, max_value=57.23, value=15.0, step=1.0)
        comp_2 = st.number_input("Comp 2", min_value=19.9, max_value=349.9, value=200.0, step=1.0)
        ps2 = st.number_input("PS2", min_value=3.3, max_value=4.4, value=4.0, step=0.1)
        fp2 = st.number_input("FP2", min_value=4.41, max_value=57.23, value=20.0, step=1.0)
        comp_3 = st.number_input("Comp 3", min_value=19.9, max_value=255.61, value=50.0, step=1.0)
        ps3 = st.number_input("PS3", min_value=3.5, max_value=4.4, value=4.0, step=0.1)
        fp3 = st.number_input("FP3", min_value=7.67, max_value=57.23, value=15.0, step=1.0)
        lag_price = st.number_input("Lag Price", min_value=19.85, max_value=364.0, value=40.0, step=1.0)

    data = {
        'product_category_name': product_category_name,
        'total_price': total_price,
        'freight_price': freight_price,
        'unit_price': unit_price,
        'product_name_lenght': product_name_lenght,
        'product_description_lenght': product_description_lenght,
        'product_photos_qty': product_photos_qty,
        'product_weight_g': product_weight_g,
        'product_score': product_score,
        'customers': customers,
        'weekday': weekday,
        'weekend': weekend,
        'holiday': holiday,
        'month': month,
        'year': year,
        's': s,
        'volume': volume,
        'comp_1': comp_1,
        'ps1': ps1,
        'fp1': fp1,
        'comp_2': comp_2,
        'ps2': ps2,
        'fp2': fp2,
        'comp_3': comp_3,
        'ps3': ps3,
        'fp3': fp3,
        'lag_price': lag_price
    }

    df = pd.DataFrame(data, index=[0])
    df['product_category_name'] = df['product_category_name'].map(category_map)
    df['log_price'] = np.log(df['unit_price'])
    return df

# Run input
df_input = user_input_features()

# Scale
scaled_input = scaler.transform(df_input)

# Predict
if st.button("🚀 Predict Quantity"):
    prediction = model.predict(scaled_input)
    st.success(f"📦 Predicted Quantity: **{prediction[0]:.2f}**")

# Footer
st.markdown("---")
st.markdown("#### 📘 About")
st.markdown("This app predicts the quantity of products sold based on various features such as price, description, photos, and competitor info.")

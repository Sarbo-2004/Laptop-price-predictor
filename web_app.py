import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load models and data
pipe = pickle.load(open('price_predict.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))  # Dataset for dropdowns

df_recommender = pickle.load(open('df_recommender.pkl', 'rb'))  # Should be a DataFrame or converted to one
price_data_raw = pickle.load(open('price_data_recommender.pkl', 'rb'))  # Could be ndarray or DataFrame
knn = pickle.load(open('knn.pkl', 'rb'))

# Convert price_data_raw to DataFrame if needed
if isinstance(price_data_raw, np.ndarray):
    # Assuming price_data_raw is a 2D ndarray with columns like [price] or [Laptop_Name, Price]
    if price_data_raw.shape[1] == 1:
        # Only prices, create DataFrame with price column only
        price_data = pd.DataFrame(price_data_raw, columns=["Price"])
    elif price_data_raw.shape[1] >= 2:
        # At least two columns, assume first is name, second price
        price_data = pd.DataFrame(price_data_raw[:, :2], columns=["Laptop_Name", "Price"])
    else:
        st.error("Unexpected shape of price data")
        price_data = pd.DataFrame()
else:
    price_data = price_data_raw

# Make sure df_recommender is a DataFrame
if not isinstance(df_recommender, pd.DataFrame):
    df_recommender = pd.DataFrame(df_recommender)

# Streamlit config
st.set_page_config(page_title="Laptop Tool", layout="centered")
st.title("💻 Laptop Price Prediction & Recommendation App")

# ------------------- Price Prediction -------------------
def predict_price():
    st.header("💰 Predict Laptop Price")

    col1, col2 = st.columns(2)
    with col1:
        company = st.selectbox("Brand", sorted(df['Company'].unique()))
        typename = st.selectbox("Type", sorted(df['TypeName'].unique()))
        ram = st.selectbox("RAM (in GB)", sorted(df['Ram'].unique()))
        weight = st.slider("Weight (kg)", 1.0, 4.0, 2.0)
        hdd = st.selectbox("HDD (in GB)", sorted(df['HDD'].unique()))
        ssd = st.selectbox("SSD (in GB)", sorted(df['SSD'].unique()))

    with col2:
        touchscreen = st.checkbox("Touchscreen")
        ips = st.checkbox("IPS Display")
        screen_size = st.slider("Screen Size (inches)", 11.0, 18.0, 15.6)
        resolution = st.selectbox("Screen Resolution", [
            '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800',
            '2880x1800', '2560x1600', '2560x1440', '2304x1440'
        ])
        cpu = st.selectbox("CPU Brand", sorted(df['Cpu brand'].unique()))
        gpu = st.selectbox("GPU Brand", sorted(df['Gpu Brand'].unique()))
        os = st.selectbox("Operating System", sorted(df['os'].unique()))

    try:
        x_res, y_res = map(int, resolution.split('x'))
        ppi = ((x_res ** 2 + y_res ** 2) ** 0.5) / screen_size
    except Exception as e:
        st.error(f"Resolution error: {e}")
        return

    touchscreen = int(touchscreen)
    ips = int(ips)

    try:
        input_df = pd.DataFrame([{
            'Company': company,
            'TypeName': typename,
            'Ram': ram,
            'Weight': weight,
            'TouchScreen': touchscreen,
            'Ips': ips,
            'ppi': ppi,
            'Cpu brand': cpu,
            'HDD': hdd,
            'SSD': ssd,
            'Gpu Brand': gpu,
            'os': os
        }])

        predicted_price = int(np.exp(pipe.predict(input_df)[0]))
        st.success(f"💵 Estimated Laptop Price: ₹ {predicted_price}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ------------------- Recommendation -------------------
def recommend_laptops():
    st.header("🎯 Recommend Laptops Based on Budget")

    budget = st.number_input("Enter your budget (in ₹)", min_value=10000, max_value=300000, value=50000, step=5000)

    try:
        distances, indices = knn.kneighbors(np.array([[budget]]))
        st.subheader("🧠 Top Laptop Picks:")

        # For each recommended laptop index, get name from df_recommender, price from price_data
        for idx in indices[0]:
            # Defensive check on index range
            if idx < len(df_recommender) and idx < len(price_data):
                # Get laptop name (adjust column name if different)
                if 'Laptop_Name' in df_recommender.columns:
                    name = df_recommender.loc[idx, 'Laptop_Name']
                else:
                    # fallback: concatenate Company + TypeName
                    name = f"{df_recommender.loc[idx, 'Company']} {df_recommender.loc[idx, 'TypeName']}"

                # Get price safely
                if 'Price' in price_data.columns:
                    price = price_data.loc[idx, 'Price']
                else:
                    price = "N/A"

                st.markdown(f"**{name}** — ₹ {price}")
            else:
                st.markdown(f"Recommendation data missing for index {idx}")
    except Exception as e:
        st.error(f"Recommendation failed: {e}")

# ------------------- Main -------------------
def main():
    st.sidebar.title("🎯 Choose Feature")
    option = st.sidebar.radio("", ["Predict Price", "Recommend Laptops"])

    if option == "Predict Price":
        predict_price()
    elif option == "Recommend Laptops":
        recommend_laptops()

if __name__ == '__main__':
    main()

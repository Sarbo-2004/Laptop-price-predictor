import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load saved models and data
pipe = pickle.load(open('price_predict.sav', 'rb'))
df_reco = pickle.load(open('df_recommender.sav', 'rb'))
price_data = pickle.load(open('price_data_recommender.sav', 'rb'))
knn = pickle.load(open('knn_model.pkl', 'rb'))
df_ui = pd.read_csv('laptop_data.csv')

st.set_page_config(page_title="Laptop Tool", layout="centered")
st.title(" Laptop Price Prediction & Recommendation App ")

# ----------------- Price Prediction -------------------
def predict_price():
    st.header(" Predict Laptop Price")

    # Input fields
    col1, col2 = st.columns(2)
    with col1:
        company = st.selectbox("Brand", sorted(df_ui['Company'].unique()))
        typename = st.selectbox("Type", sorted(df_ui['TypeName'].unique()))
        ram = st.selectbox("RAM (in GB)", sorted(df_ui['Ram'].unique()))
        weight = st.slider("Weight (kg)", 1.0, 4.0, 2.0)

    with col2:
        touchscreen = st.checkbox("Touchscreen")
        ips = st.checkbox("IPS Display")
        screen_size = st.slider("Screen Size (inches)", 11.0, 18.0, 15.6)
        resolution = st.selectbox("Screen Resolution", [
            '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800',
            '2880x1800', '2560x1600', '2560x1440', '2304x1440'
        ])
        cpu = st.selectbox("CPU Brand", sorted(df_ui['Cpu'].unique()))

    # Calculate PPI
    try:
        x_res, y_res = map(int, resolution.split('x'))
        ppi = ((x_res**2 + y_res**2)**0.5) / screen_size
    except Exception as e:
        st.error(f"Resolution error: {e}")
        return

    # Prepare input for prediction
    query = np.array([company, typename, ram, weight, touchscreen, ips, ppi, cpu])
    query = query.reshape(1, -1)

    # Predict
    try:
        predicted_price = int(np.exp(pipe.predict(query)[0]))
        st.success(f"Estimated Laptop Price: ₹ {predicted_price}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ----------------- Laptop Recommender -------------------
def recommend_laptops():
    st.header(" Recommend Laptops Based on Budget ")

    budget = st.number_input("Enter your budget (in ₹)", min_value=10000, max_value=300000, value=50000, step=5000)

    try:
        distances, indices = knn.kneighbors([[budget]])
        st.subheader("Top Laptop Picks for You:")
        for i in indices[0]:
            laptop_name = df_reco.iloc[i]['Laptop_Name']
            price = price_data.iloc[i]['Price']
            st.write(f"**{laptop_name}** — ₹ {price}")
    except Exception as e:
        st.error(f"Recommendation failed: {e}")

# ----------------- Main Function -------------------
def main():
    st.sidebar.title(" Choose Feature ")
    option = st.sidebar.radio("", ["Predict Price", "Recommend Laptops"])

    if option == "Predict Price":
        predict_price()
    elif option == "Recommend Laptops":
        recommend_laptops()

if __name__ == '__main__':
    main()

import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Function to calculate CLV using Linear Regression
def calculate_clv(data):
    X = data[['recency', 'frequency', 'monetary']]
    y = data['clv']

    model = LinearRegression()
    model.fit(X, y)

    return model

# Function to generate a synthetic dataset
def generate_synthetic_data():
    np.random.seed(42)
    customer_ids = ['C' + str(i).zfill(5) for i in range(1, 21)]
    recency = np.random.randint(10, 365, size=20)
    frequency = np.random.randint(1, 10, size=20)
    monetary = np.random.randint(20, 200, size=20)
    clv = np.random.randint(100, 1000, size=20)

    data = pd.DataFrame({
        'customer_id': customer_ids,
        'recency': recency,
        'frequency': frequency,
        'monetary': monetary,
        'clv': clv
    })

    return data

# Main function to run the Streamlit app
def main():
    st.title('Customer Lifetime Value Prediction')
    st.write('Generating synthetic dataset...')
    data = generate_synthetic_data()

    st.dataframe(data.head())

    model = calculate_clv(data)

    st.write('Model trained! Enter customer data below to predict their CLV.')
    recency = st.number_input('Recency (days since last purchase):', value=30)
    frequency = st.number_input('Frequency (number of purchases in the last year):', value=3)
    monetary = st.number_input('Monetary (average purchase amount in the last year):', value=50)

    if st.button('Predict CLV'):
        input_data = pd.DataFrame({'recency': [recency], 'frequency': [frequency], 'monetary': [monetary]})
        clv_prediction = model.predict(input_data)
        st.success(f'Predicted CLV: ${clv_prediction[0]:.2f}')

if __name__ == '__main__':
    main()

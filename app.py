# Import necessary libraries
import streamlit as st 
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Preprocess the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop(columns=['target']))
y = df['target']

# Train a simple ANN model
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
model.fit(X_scaled, y)

# Streamlit Page Configuration
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# App Title and Description
st.title("🎗️ Breast Cancer Prediction")
st.write(
    """
    This application predicts whether a breast tumor is **malignant** or **benign** based on diagnostic features. 
    Adjust the feature sliders and click 'Predict' to get the result. 🚀
    """
)

# Sidebar for user input (input sliders)
st.sidebar.header("📝 Enter Feature Values")
input_data = []
for feature in data.feature_names:
    value = st.sidebar.slider(
        feature,
        min_value=float(df[feature].min()),
        max_value=float(df[feature].max()),
        value=float(df[feature].mean()),
        step=0.01,
    )
    input_data.append(value)

# Convert user inputs into DataFrame
input_df = pd.DataFrame([input_data], columns=data.feature_names)
input_scaled = scaler.transform(input_df)  # Scale the input data

# Prediction Button
st.markdown("---")
predict_button = st.button("🔮 Predict")

# Prediction and Visualization
if predict_button:
    st.markdown("---")
    st.subheader("🔮 Prediction Result")

    # Make the prediction
    prediction = model.predict(input_scaled)
    result = "Malignant 🩸" if prediction[0] == 0 else "Benign 🟢"

    # Display result with dynamic styling
    if prediction[0] == 0:
        st.error(f"Prediction: {result}", icon="💔")
    else:
        st.success(f"Prediction: {result}", icon="🩺")

    # Display the input values
    st.subheader("📋 Input Feature Values")
    st.dataframe(input_df.T.rename(columns={0: "Value"}))

    # Visualize input data
    st.subheader("📈 Feature Value Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=input_df.columns, y=input_df.iloc[0], ax=ax, color="teal")
    plt.xticks(rotation=90)
    ax.set_xlabel("Features")
    ax.set_ylabel("Value")
    ax.set_title("Input Feature Values")
    plt.tight_layout()
    st.pyplot(fig)

# Footer
st.markdown("---")  # Add a horizontal line

st.info(
    """
    💡 **Pro Tip:** Adjust the sliders to see how different values of features affect the prediction.
    This app is for educational purposes only and should not replace professional medical advice.
    """
)


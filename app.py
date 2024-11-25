# Import necessary libraries
import streamlit as st  # For creating the web application
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
from sklearn.datasets import load_breast_cancer  # Load the Breast Cancer dataset
from sklearn.neural_network import MLPClassifier  # Artificial Neural Network (ANN) for classification
from sklearn.preprocessing import StandardScaler  # For feature scaling
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For advanced data visualization

# Load the Breast Cancer dataset
data = load_breast_cancer()  # Loads a dataset containing diagnostic features and target labels
df = pd.DataFrame(data.data, columns=data.feature_names)  # Converts the data into a pandas DataFrame
df['target'] = data.target  # Adds the target column (0 = malignant, 1 = benign)

# Preprocess the dataset
scaler = StandardScaler()  # Initialize a StandardScaler for feature scaling
X_scaled = scaler.fit_transform(df.drop(columns=['target']))  # Scale the feature data
y = df['target']  # Extract the target variable

# Train a simple Artificial Neural Network (ANN) model
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)  # ANN with 100 hidden neurons
model.fit(X_scaled, y)  # Fit the model on scaled features and target labels

# Streamlit Page Configuration
st.set_page_config(
    page_title="Breast Cancer Predictor",  # Sets the title of the app
    page_icon="🎗️",  # Sets a ribbon emoji as the page icon
    layout="wide",  # Sets the layout to wide mode for better visualization
    initial_sidebar_state="collapsed",  # Collapses the sidebar by default
)

# App Title and Description
st.title("🎗️ Breast Cancer Prediction")  # Main title of the app
st.write(
    """
    This application predicts whether a breast tumor is **malignant** or **benign** based on diagnostic features. 
    Adjust the feature sliders and click 'Predict' to get the result. 🚀
    """
)  # Brief description of the app

# Sidebar for user input (feature sliders)
st.sidebar.header("📝 Enter Feature Values")  # Header for sidebar
input_data = []  # Initialize a list to store user inputs
for feature in data.feature_names:  # Loop through each feature in the dataset
    value = st.sidebar.slider(  # Create a slider for each feature
        feature,
        min_value=float(df[feature].min()),  # Minimum value of the slider
        max_value=float(df[feature].max()),  # Maximum value of the slider
        value=float(df[feature].mean()),  # Default value (mean of the feature)
        step=0.01,  # Step size for the slider
    )
    input_data.append(value)  # Append the slider value to the input list

# Convert user inputs into a DataFrame
input_df = pd.DataFrame([input_data], columns=data.feature_names)  # Create a DataFrame with user inputs
input_scaled = scaler.transform(input_df)  # Scale the input data using the same scaler as training data

# Prediction Button
st.markdown("---")  # Add a horizontal line for separation
predict_button = st.button("🔮 Predict")  # Create a button for making predictions

# Prediction and Visualization
if predict_button:  # When the Predict button is clicked
    st.markdown("---")  # Add another horizontal line for separation
    st.subheader("🔮 Prediction Result")  # Subheader for the prediction result

    # Make the prediction
    prediction = model.predict(input_scaled)  # Use the trained model to make predictions
    result = "Malignant 🩸" if prediction[0] == 0 else "Benign 🟢"  # Map prediction to result text

    # Display result with appropriate styling
    if prediction[0] == 0:  # If malignant
        st.error(f"Prediction: {result}", icon="💔")  # Display an error message
    else:  # If benign
        st.success(f"Prediction: {result}", icon="🩺")  # Display a success message

    # Display the input values
    st.subheader("📋 Input Feature Values")  # Subheader for input values
    st.dataframe(input_df.T.rename(columns={0: "Value"}))  # Transpose and display the input data

    # Visualize input data
    st.subheader("📈 Feature Value Visualization")  # Subheader for visualization
    fig, ax = plt.subplots(figsize=(10, 6))  # Create a plot with custom dimensions
    sns.barplot(x=input_df.columns, y=input_df.iloc[0], ax=ax, color="teal")  # Bar plot of feature values
    plt.xticks(rotation=90)  # Rotate x-axis labels for readability
    ax.set_xlabel("Features")  # X-axis label
    ax.set_ylabel("Value")  # Y-axis label
    ax.set_title("Input Feature Values")  # Plot title
    plt.tight_layout()  # Adjust layout to avoid overlap
    st.pyplot(fig)  # Display the plot in the app

# Footer
st.markdown("---")  # Add a horizontal line
st.info(
    """
    💡 **Pro Tip:** Adjust the sliders to see how different values of features affect the prediction.
    This app is for educational purposes only and should not replace professional medical advice.
    """
)  


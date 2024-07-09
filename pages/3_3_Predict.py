import streamlit as st
import json
import os
import pickle
import pandas as pd

# Function to read selected features and their data types from the JSON file
def load_selected_features(file_path):
    with open(file_path, 'r') as f:
        selected_features_with_types = json.load(f)
    return selected_features_with_types

# Function to create input fields based on data types
def create_input_fields(features_with_types):
    user_inputs = {}
    for feature, dtype in features_with_types.items():
        if dtype == 'int64':
            user_inputs[feature] = st.sidebar.checkbox(f'{feature}', value=False, key=feature)
        elif dtype == 'float64':
            user_inputs[feature] = st.sidebar.number_input(f'Enter value for {feature}', value=0.0)
        else:
            st.sidebar.write(f'Unsupported data type {dtype} for feature {feature}')
    return user_inputs

# Function to load all models from the models folder
def load_models(models_folder):
    models = {}
    for model_file in os.listdir(models_folder):
        if model_file.endswith('.pkl'):
            with open(os.path.join(models_folder, model_file), 'rb') as f:
                models[model_file] = pickle.load(f)
    return models

# Load models from the models folder
models_folder = 'models/'
models = load_models(models_folder)

# Check if models were loaded
if not models:
    st.title("Model Predictions")
    st.warning("🚨 No models are loaded. Please go to Step 2 and train a model before making predictions.")
else:
    # Load selected features and their data types
    selected_features_with_types = load_selected_features('parameters/selected_features.json')
    
    # Create input fields and collect user inputs
    st.sidebar.title("Input Features")
    user_inputs = create_input_fields(selected_features_with_types)
    
    # Function to predict using each model
    def predict_with_models(models, user_inputs):
        predictions = {}
        input_data = pd.DataFrame([user_inputs])
        for model_name, model in models.items():
            predictions[model_name] = model.predict(input_data)[0]  # Assuming the model has a predict method
        return predictions

    # Predict with each model
    predictions = predict_with_models(models, user_inputs)

    # Create a DataFrame to display model names and predictions
    predictions_df = pd.DataFrame(list(predictions.items()), columns=['Model', 'Prediction'])

    # Display the predictions in a styled DataFrame
    st.title("Model Predictions")
    st.write("This app predicts outcomes based on selected features and various models.")
    st.dataframe(predictions_df)

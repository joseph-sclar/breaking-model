import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, make_scorer
import pandas as pd
import pickle
import os
import altair as alt
import json


st.title("Train Model")



tab1, tab2,  tab3 = st.tabs(["Feature Selection", "Model Training", "Model Evaluation"])


with tab1:
    
    def delete_files_in_directory(directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    st.header("Feature Selection")
    st.write("This section allows you to select features for your model.")
    st.write("Here we are choosing the features based on the Exploratory Data Analysis (EDA) that we think are the most relevant for predicting our target variable. EDA helps in understanding the data better, identifying important patterns, and selecting the features that have the most predictive power. By carefully choosing these features, we aim to improve the performance and accuracy of our machine learning model.")
    st.write("Feel free to use the default features provided, or change them to try and make the model more accurate. Experimenting with different feature combinations can help in finding the optimal set for your specific use case.")
    
    
    st.write("********************************")
    
    st.subheader("Select Features")
    # Load encoded data
    data = pd.read_csv("data/data_encoded.csv")
    data_encoded = data.select_dtypes(include=['int', 'float', 'bool'])

    # List of column names
    columns = data_encoded.columns.tolist()
    if 'has_potential' in columns:
        columns.pop(columns.index('has_potential'))

    # Define the default columns
    default_columns = ['avg_monthly_onsite_sales', 'avg_monthly_uvs', 'CPO_Retargeting', 'landing_rate', 'sales']

    # Path to the parameters folder
    params_folder = 'parameters/'
    selected_features_file = os.path.join(params_folder, 'selected_features.json')

    # Check if the folder is empty
    if not os.listdir(params_folder):
        columns_to_use = default_columns
    else:
        # Check if the selected_features.json file exists
        if os.path.isfile(selected_features_file):
            with open(selected_features_file, 'r') as file:
                data = json.load(file)
                columns_to_use = list(data.keys())  # Use the keys from the JSON file as columns
        else:
            # If the file does not exist, use the default columns
            columns_to_use = default_columns


    # Select features
    selected_features = st.multiselect(
        "Select Features to train Model",
        columns,
        default=columns_to_use,
        disabled=False,
        label_visibility="visible",
        key='selected_features',
        on_change=lambda: delete_files_in_directory('models/')
    )

    selected_features_with_types = {feature: str(data_encoded[feature].dtype) for feature in selected_features}

    # Save the selected features along with their data types to a JSON file
    with open('parameters/selected_features.json', 'w') as f:
        json.dump(selected_features_with_types, f)

    # Display selected features and data types:
    st.write("**Selected Features:**")
    for i in range(0, len(selected_features)):
        st.write(selected_features[i], data_encoded[selected_features[i]].dtype)
        
    
    
    

with tab2:
    st.header("Model Training")
    st.write("This section allows you to train your model.")
    st.write("Here, we will train different algorithms to see which one performs better. We'll use a method called GridSearch to try different parameters and find the best combination for the model.")
    st.write("We will focus on optimizing for precision, which means we want to minimize false positives as much as possible.")
    
    
    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs(["Algorithm Selection", "Hyperparameters", "Data Split", "Start Training"])
   
    with sub_tab1:
        with st.expander("Algorithms", expanded=True):
            st.subheader("Algorithm Selection")
            st.write("Select the algorithms you want to use to train your model.")
            
            algorithms = st.multiselect("Select Algorithms", ["Logistic Regression", "Decision Tree", "Random Forest", 'SVC', 'KNN'], default=["Logistic Regression", "Decision Tree", "Random Forest"])
            
            if not algorithms:
                st.error("Please select at least one algorithm.")
            


    with sub_tab2:

        st.subheader("Hyperparameter Selection")
        st.write("Select hyperparameters for the chosen algorithms.")
        st.write("Hyperparameters are settings that you can adjust to improve how your model works. Different settings can lead to better or worse results.")
        st.write("We will use GridSearch, which is a method that tries out different combinations of these settings to find the best ones for your model. This helps us make sure we're getting the best possible performance from our model.")
        
        if not algorithms:
            st.warning("Please go to the Algorithm Selection tab and choose at least one algorithm.")
        else:
            hyperparams = {}

        for algorithm in algorithms:
            with st.expander(f"Hyperparameters for {algorithm}"):
                if algorithm == "Logistic Regression":
                    max_iter = st.multiselect("Max Iterations", options=[100, 200, 300], default=[100], key=f"lr_max_iter_{algorithm}")
                    penalty = st.multiselect("Penalty", options=['l1', 'l2', 'elasticnet', 'none'], default=['l2'], key=f"lr_penalty_{algorithm}")
                    hyperparams["Logistic Regression"] = {'max_iter': max_iter, 'penalty': penalty}
                elif algorithm == "Decision Tree":
                    max_depth = st.multiselect("Max Depth", options=list(range(1, 11)), default=[5], key=f"dt_max_depth_{algorithm}")
                    min_samples_split = st.multiselect("Min Samples Split", options=list(range(2, 6)), default=[2], key=f"dt_min_samples_split_{algorithm}")
                    criterion = st.multiselect("Criterion", options=['gini', 'entropy'], default=['gini'], key=f"dt_criterion_{algorithm}")
                    hyperparams["Decision Tree"] = {'max_depth': max_depth, 'min_samples_split': min_samples_split, 'criterion': criterion}
                elif algorithm == "Random Forest":
                    n_estimators = st.multiselect("Number of Trees", options=[50, 100, 200], default=[100], key=f"rf_n_estimators_{algorithm}")
                    max_depth = st.multiselect("Max Depth", options=list(range(1, 11)), default=[5], key=f"rf_max_depth_{algorithm}")
                    min_samples_split = st.multiselect("Min Samples Split", options=list(range(2, 6)), default=[2], key=f"rf_min_samples_split_{algorithm}")
                    hyperparams["Random Forest"] = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_split': min_samples_split}
                elif algorithm == "SVC":
                    C = st.multiselect("C (Regularization Parameter)", options=[0.1, 1.0, 10.0], default=[1.0], key=f"svc_c_{algorithm}")
                    kernel = st.multiselect("Kernel", options=['linear', 'poly', 'rbf'], default=['rbf'], key=f"svc_kernel_{algorithm}")
                    gamma = st.multiselect("Gamma", options=['scale', 'auto'], default=['scale'], key=f"svc_gamma_{algorithm}")
                    hyperparams["SVC"] = {'C': C, 'kernel': kernel, 'gamma': gamma}
                elif algorithm == "KNN":
                    n_neighbors = st.multiselect("Number of Neighbors", options=list(range(1, 11)), default=[5], key=f"knn_n_neighbors_{algorithm}")
                    algorithm_knn = st.multiselect("Algorithm", options=['auto', 'ball_tree', 'kd_tree', 'brute'], default=['auto'], key=f"knn_algorithm_{algorithm}")
                    hyperparams["KNN"] = {'n_neighbors': n_neighbors, 'algorithm': algorithm_knn}

        


        
        with sub_tab3:
            with st.expander("Data Split", expanded=True):
                st.subheader("Split Data")
                st.write("Select the split ratio for training and testing data. The training data is used to train your model, while the testing data is used to evaluate its performance.")
                st.write("A common split ratio is 80:20, meaning 80% of the data is used for training and 20% is used for testing.")
                st.write("**Recommended:** Use an 80:20 or 70:30 split ratio for balanced training and evaluation.")

                split_ratio = st.select_slider("Train-Test Split Ratio", options=[0, 0.1, 0.2, 0.3, 0.4, 0.5, .6, .7, .8, .9, 1], value=0.8, key='split_ratio')
                
                if 'data_encoded' in globals():
                    st.write("Train: {:,}\nTest: {:,}".format(int(split_ratio * len(data_encoded)), int((1 - split_ratio) * len(data_encoded))))
                
                X_train, X_test, y_train, y_test = train_test_split(data_encoded[selected_features], data_encoded['has_potential'], test_size=1-split_ratio, random_state=42)
            
        with sub_tab4:

            with st.expander("Grid Search Setup", expanded=True):
                st.subheader("Grid Search Setup")
                st.write("Set up the grid search for hyperparameter tuning of the selected models.")
                
                if not algorithms:
                    st.warning("Please go to the Algorithm Selection tab and choose at least one algorithm.")
                else:
                    st.write("The following grid search will be performed based on your selected algorithms and hyperparameters:")
                    grid_search_params = {}
                    
                    selected_features = selected_features.copy()
                    
                    st.write("Independent Features:")
                    st.write(selected_features)
                    
                    st.write("Grid Search Parameters:")
                    
                    
                    
                    for algorithm in algorithms:
                        grid_search_params[algorithm] = hyperparams[algorithm]
                        st.code(f'{algorithm}: {hyperparams[algorithm]}')
                    
                    if st.button("Start Grid Search"):
                        with st.spinner('Performing grid search... this may take a while.'):
                            

                            results = {}
                            for algorithm in algorithms:
                                st.markdown(f"- Performing grid search for **{algorithm}**...")
                                if algorithm == "Logistic Regression":
                                    model = LogisticRegression()
                                elif algorithm == "Decision Tree":
                                    model = DecisionTreeClassifier()
                                elif algorithm == "Random Forest":
                                    model = RandomForestClassifier()
                                elif algorithm == "SVC":
                                    model = SVC()
                                elif algorithm == "KNN":
                                    model = KNeighborsClassifier()
                                
                                grid_search = GridSearchCV(estimator=model, param_grid=grid_search_params[algorithm], cv=5, n_jobs=-1, scoring=make_scorer(precision_score, pos_label=1), verbose=2)
                                grid_search.fit(X_train, y_train)
                                
                                best_params = grid_search.best_params_
                                best_score = grid_search.best_score_
                                results[algorithm] = {'best_params': best_params, 'best_score': best_score}
                                
                                # Save the best model
                                best_model = grid_search.best_estimator_
                                with open(f'models/best_{algorithm.replace(" ", "_")}.pkl', 'wb') as f:
                                    pickle.dump(best_model, f)
                        
                        st.success("Grid search completed. Here are the results:")
                        for algorithm in results:
                            st.write(f"**{algorithm}**:")
                            st.json({
                                "Best Parameters": results[algorithm]['best_params'],
                                "Best Precision Score": results[algorithm]['best_score']
                            })
                        st.write("All best models have been saved.")

    # Ensure the models directory exists
    os.makedirs('models', exist_ok=True)




    # Load the saved models for evaluation
    model_files = [f'models/{file}' for file in os.listdir('models') if file.endswith('.pkl')]
    models = {file.split('/')[-1].replace('best_', '').replace('.pkl', '').replace('_', ' '): pickle.load(open(file, 'rb')) for file in model_files}

    # Initialize an empty list to store evaluation results
    evaluation_results = []

    # Perform evaluation on each model
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        evaluation_results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "True Positives": tp,
            "False Positives": fp
        })

        # Debugging print statements
        print(f"Confusion Matrix for {model_name}:")
        print(confusion_matrix(y_test, y_pred))

    # Convert the evaluation results into a DataFrame
    evaluation_df = pd.DataFrame(evaluation_results)


with tab3:
    

    st.header("Model Evaluation")
    st.write("This section allows you to evaluate your model.")

    subtab_model, subtab_matrix = st.tabs(['Model Results', 'Confusion Matrices'])

    with subtab_model:
        
        st.write("""
        ### Understanding Model Performance

        1. **Key Terms**:
            - **Accuracy**: How often the model is correct overall.
            - **Precision**: Out of all the campaigns predicted to be successful, how many actually are. Important to minimize mistakes.
            - **Recall**: Out of all the actual successful campaigns, how many did the model correctly identify.
            - **F1 Score**: A balance between precision and recall.
            - **True Positives (TP)**: The number of correct successful campaign predictions.
            - **False Positives (FP)**: The number of incorrect successful campaign predictions.
            

        2. **Our Goal**:
            - We want to identify campaigns that are likely to be successful so we can cross-sell services. High precision is key for this.


        3. **Steps to Follow**:
            - **Check Precision**: We look for models with high precision to ensure we are recommending likely successful campaigns.
            - **Review True Positives**: We make sure the model still identifies a good number of successful campaigns.


        4. **Decision Making**:
            - We pick the model with the best combination of high precision and a reasonable number of successful campaigns (true positives).
            - This approach helps us confidently identify campaigns that are likely to succeed, ensuring we offer our service effectively.
        """)
        
        st.write('---')
        
        st.dataframe(evaluation_df)
    
    with subtab_matrix:
    
        st.write("""
        ### Understanding Confusion Matrices

        A confusion matrix helps us understand the performance of our classification models by showing the actual versus predicted outcomes.

        1. **Components of a Confusion Matrix**:
        - **True Positives (TP)**: The model correctly predicts a successful campaign.
        - **True Negatives (TN)**: The model correctly predicts an unsuccessful campaign.
        - **False Positives (FP)**: The model incorrectly predicts a campaign will be successful (Type I error).
        - **False Negatives (FN)**: The model incorrectly predicts a campaign will not be successful (Type II error).

        2. **Interpreting the Confusion Matrix**:
        - **High True Positives and True Negatives**: Indicates good model performance.
        - **Low False Positives**: Important for us because we want to minimize recommending unsuccessful campaigns.
        - **Low False Negatives**: Indicates fewer missed opportunities for recommending successful campaigns.

        3. **Our Goal**:
        - Focus on models with a high number of true positives and low false positives to ensure accurate recommendations.
        """)
        
        
        
        st.write('---')
        
        st.subheader("Confusion Matrices")

        # Display confusion matrices in columns of 2
        cols = st.columns(2)
        col_index = 0
        
        

        for model_name, model in models.items():
            with cols[col_index]:
                st.write(f"Confusion Matrix for {model_name}")
                y_pred = model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                cm_df = pd.DataFrame(cm, index=["Negative", "Positive"], columns=["Negative", "Positive"])

                # Melt the dataframe for Altair
                cm_df = cm_df.reset_index().melt(id_vars='index')
                cm_df.columns = ['Actual', 'Predicted', 'Count']

                # Create the confusion matrix heatmap using Altair
                chart = alt.Chart(cm_df).mark_rect().encode(
                    x='Predicted:O',
                    y='Actual:O',
                    color=alt.Color('Count:Q', scale=alt.Scale(scheme='blueorange'),  legend=None), 
                    tooltip=['Actual', 'Predicted', 'Count']
                ).properties(
                    width=300,
                    height=300
                ) + alt.Chart(cm_df).mark_text(size=20).encode(
                    x='Predicted:O',
                    y='Actual:O',
                    text='Count:Q',
                    color=alt.condition(
                        alt.datum.Count > cm_df['Count'].max() / 2,
                        alt.value('white'),
                        alt.value('black')
                    )
                )

                st.altair_chart(chart, use_container_width=True)
            
            # Move to the next column or reset to the first column
            col_index = (col_index + 1) % 2


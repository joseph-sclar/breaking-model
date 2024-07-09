import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import altair as alt
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv('data/data.csv')



# Handle missing values
data['daily_budget_euro'].fillna(data['daily_budget_euro'].median(), inplace=True)

# One-hot encode categorical variables
categorical_cols = ['campaign_type', 'channel', 'industry', 'region', 'business_model', 'context']

data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=False)

# Ensure 'has_potential' remains as boolean
data_encoded['has_potential'] = data['has_potential'].astype(bool)

# Standardize numerical features except 'has_potential'
numerical_cols = data_encoded.select_dtypes(include=['float64', 'int64', 'int32']).columns

robust_scaler = RobustScaler()
data_encoded[numerical_cols] = robust_scaler.fit_transform(data_encoded[numerical_cols])

data_encoded.to_csv("data/data_encoded.csv", index=False)

correlation_df = data_encoded[numerical_cols].copy()
correlation_df['has_potential'] = data_encoded['has_potential']

# Correlation matrix
correlation_matrix = correlation_df.corr()
target_correlation = correlation_matrix['has_potential'].sort_values(ascending=False)

# Random Forest for feature importance
y = data_encoded['has_potential']
X = data_encoded.drop(['has_potential', 'client_name', 'campaign_id'], axis=1)

model = RandomForestClassifier()
model.fit(X, y)
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

# Streamlit Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Data Preparation", "Correlation", "Feature Importance", "Distributions"])

with tab1:
    st.header("Dataset Overview")
    st.write("This section provides an initial overview of the dataset. It includes a preview of the data and basic statistics.")
    st.write(data)

with tab2:
    st.header("Data Preparation")
    st.write("This section covers handling missing values, one-hot encoding of categorical variables, and feature scaling.")
    
    st.write("---")
    
    st.subheader("Missing Values")
    st.write("We identify and handle missing values in the dataset to ensure our data is clean for analysis.")
    st.dataframe(data.isnull().sum())
    
    st.write("---")
    
    st.subheader("One-Hot Encoding")
    st.write("We convert categorical variables into a format that can be provided to ML algorithms to improve predictions.")
    st.write(data_encoded)
    
    st.write("---")
    
    st.subheader("Feature Scaling")
    st.write("We standardize numerical features to ensure they are on a similar scale, which can improve the performance and training stability of the model.")
    st.write(data_encoded[numerical_cols])

with tab3:
    

    correlation_matrix = correlation_matrix.reset_index().rename(columns={'index': 'feature'}).melt(id_vars='feature')
    
    
    full_heatmap = alt.Chart(correlation_matrix).mark_rect().encode(
        x=alt.X('feature:N', title=''),
        y=alt.Y('variable:N', title=''),
        color=alt.Color('value:Q', scale=alt.Scale(scheme='redblue'), legend=None),
        tooltip=['feature', 'variable', 'value']
    ).properties(
        width=600,
        height=600
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    )
    
    target_correlation = target_correlation.reset_index().rename(columns={'index': 'feature'}).melt(id_vars='feature')
    
    index_to_drop = target_correlation[(target_correlation['feature'] == 'has_potential')].index
    filtered_data = target_correlation.drop(index=index_to_drop)
    
    
    target_heatmap = alt.Chart(filtered_data).mark_rect().encode(
        y=alt.X('feature:N', title=''),
        x=alt.Y('variable:N', title=''),
        color=alt.Color('value:Q', scale=alt.Scale(scheme='redblue' ), legend=None),
        tooltip=['feature', 'variable', 'value']
    ).properties(
        width=350,
        height=600
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    )

    # Display in Streamlit
    st.title("Correlation Analysis")
    st.write("The correlation matrix helps us understand the relationships between different features and our target variable 'has_potential'.")
    
    st.write("---")
    
    st.header("Key Learnings")
    st.subheader("Positive Correlations")
    st.write("""
    - **Engagement & Sales:** We observed that average monthly unique visitors and onsite sales are strongly correlated. Higher engagement typically aligns with higher sales. However, they are also highly correlated with each other, suggesting potential multicollinearity. It might be beneficial to consider using only one of these variables to avoid redundancy.
    - **Budget & Cost Management:** The data suggests that cost per order for retargeting and daily budget (in euros) positively correlate with potential. This indicates that efficient budget allocation and cost management are important factors.
    """)

    st.subheader("Negative Correlations")
    st.write("""
    - **Campaign Duration & Landing Rate:** We learned that longer campaign durations and lower landing rates are negatively correlated with potential. This suggests that optimal campaign duration and higher landing rates are beneficial.
    """)

    st.subheader("Multicollinearity Concerns")
    st.write("""
    - **Days Live & Landing Rate:** There is a moderate correlation between the number of days a campaign is live and the landing rate, indicating some interrelated effects. 
    - **Other Variables:** We see that CPO Retargeting, daily budget, days live, and landing rate show low correlations with each other, suggesting no significant multicollinearity among these variables.
    """)

    st.subheader("Correlation Matrix Observations")
    st.write("""
    - **High Correlation:** The correlation matrix indicates that spend, daily budget, clicks, displays, and visits are strongly correlated. Similarly, conversion rate (CR) is highly correlated with order value and sales. These strong correlations hint at potential multicollinearity if these variables are included in the model.
    """)
    
    st.write("---")
    
    st.subheader("Target Correlation Values")
    col1, col2 = st.columns(2)
    
    
    with col1:
        
        target_correlation_disaply = filtered_data[['feature', 'value']]
        st.write(target_correlation_disaply)
        
        
    with col2:
        st.altair_chart(target_heatmap, use_container_width=True)
    
    st.write("---")
    st.subheader("Full Correlation Matrix")
    st.altair_chart(full_heatmap, use_container_width=True)
    
    
    
    


    


with tab4:
    st.header("Feature Importance")
    st.write("Using a Random Forest classifier, we determine which features are most important for predicting 'has_potential'. Random Forest is chosen because it combines the outputs of multiple decision trees to improve predictive accuracy and control over-fitting. This ensemble method helps in capturing the importance of various features by evaluating the contribution of each feature to the decision-making process across all trees.")
    
    st.write("---")
    
    st.subheader("Key Learnings")
    st.write("""
    - **Visitor and Sales Metrics:** We have learned that average monthly unique visitors and onsite sales are top indicators of campaign success.
    - **Cost Efficiency and Engagement:** The data suggest that optimizing Cost Per Click (CPC), Click-Through Rates (CTR), and audience size is crucial for balancing cost efficiency and high engagement.
    - **Secondary Factors:** The analysis indicates that campaign types, regional targeting, and industry specifics can help refine strategies, though they are less influential compared to primary metrics.
    """)
    
    st.write("---")
    
  
    feature_importances = pd.DataFrame(feature_importances).reset_index()
    feature_importances.columns = ['Feature', 'Importance']

    feature_importances['Importance'] = pd.to_numeric(feature_importances['Importance'], errors='coerce')
    
    
    # Create an Altair chart
    chart = alt.Chart(feature_importances).mark_bar().encode(
        x='Importance:Q',
        y=alt.Y('Feature:N', sort='-x'),
        color=alt.Color('Importance:Q', scale=alt.Scale(scheme='redblue' ), legend=None),
    ).properties(
        title='Feature Importances based on the Random Forest Classifier',
    )

    # Streamlit app
    st.header('Feature Importances')
    st.altair_chart(chart, use_container_width=True)



with tab5:
    st.header("Distributions")
    st.write("This section allows us to visualize the distribution of different variables, helping us understand their behavior and relationship with 'has_potential'.")
    
    columns = data.columns
    columns = columns.drop(['has_potential', 'client_name', 'campaign_id'])
    
    selected_col = st.selectbox("Select a variable to visualize its distribution:", columns, key="distribution")

    log_sclabel = st.checkbox("Use log scale", key="logscale_distribution")

    if selected_col in numerical_cols:
        data[selected_col] = data[selected_col].replace([np.inf, -np.inf], np.nan)
        
        data = data.dropna(subset=[selected_col])
        
        fig = px.box(data, x='has_potential', y=selected_col, color='has_potential',)

        fig.update_layout(
            title=f'Distribution of {selected_col}',
            xaxis=dict(title='Has Potential'),
            yaxis=dict(title=selected_col, type='log' if log_sclabel else 'linear')
        )

        # Display the plot
        st.plotly_chart(fig)
        
        st.header("What are we looking for?")
        st.write(f"""
        When checking these box plots to see if **{selected_col}** is a good feature for predicting potential, we are looking for:
        - **Different Medians**: If the middle lines in the boxes (medians) are far apart, it’s a good sign.
        - **Box Overlap**: Less overlap between the boxes means better separation between groups.
        - **Outliers**: Many outliers in one group compared to the other can indicate a difference.
        - **Whisker Length**: Big differences in the lines extending from the boxes show variability.

        In simple terms, if the two groups (has potential vs. no potential) look different in the box plot, then **{selected_col}** is likely a good feature to use for predicting potential.
        """)

    else:
        # Creating a count plot with Plotly
        fig = px.histogram(data, x=selected_col, color='has_potential', barmode='group')
        fig.update_layout(
            title=f'Distribution of {selected_col}',
            xaxis=dict(title=selected_col),
            yaxis=dict(title='Count', type='log' if log_sclabel else '-')
        )
        
        st.plotly_chart(fig)
        
        st.header("What are we looking for?")
        st.write(f"**Represenation of {selected_col}**")
        st.write(f"- When looking at the plots, we are checking if **{selected_col}** values are evenly represented. Uneven data can bias the model. Making sure each **{selected_col}** value has enough 'potential' (1) and 'non-potential' (0) cases help accurate predictions.")
        st.write(f"**Proportions of 'potential' vs. 'non-potential' cases within {selected_col}**")
            
        st.write(f" - Big proportion differences in 'potential' vs. 'non-potential' across values suggest that **{selected_col}** is a good predictor. If the proportions are similar, the **{selected_col}** might not be useful for the model. Using log scale might help us visualize better this differences. This helps us choose features for training.")

import streamlit as st



# Title of the page
st.title("Breaking Model Project")

# Intro Section
st.header("Introduction")
st.write("""
Welcome to the Breaking Model project! This project aims to identify which clients have the potential to achieve outstanding performance if they launch an acquisition campaign. 
For simplicity, the dataset and steps shown have been cleaned, modified, and simplified to ensure privacy.
""")

# Goal Section
st.header("Goal")
st.write("""
The primary goal of this project is to predict which clients would benefit the most from an acquisition campaign. 
We define 'potential' as clients having a CPO Ratio (CPO_Retargeting / CPO_Acquisition) less than 1.
""")

# CPO Ratio Explanation
st.header("Understanding the CPO Ratio")
st.write("""
The Cost Per Order (CPO) ratio is a crucial metric in evaluating the effectiveness of marketing campaigns. It is calculated as follows:

**CPO Ratio = CPO_Retargeting / CPO_Acquisition**

- **CPO_Retargeting**: The cost incurred per order through retargeting efforts.
- **CPO_Acquisition**: The cost incurred per order through acquisition efforts.

A CPO Ratio of less than 1 indicates that acquisition efforts are more cost-effective than retargeting efforts for a particular client. This metric helps us identify clients who would likely achieve better performance through acquisition campaigns rather than retargeting.
""")

# Importance of the CPO Ratio
st.write("""
### Why is the CPO Ratio Important?
The CPO Ratio is essential because it highlights clients who have the potential to achieve outstanding performance through acquisition campaigns. This presents a significant opportunity for the company to cross-sell acquisition campaigns as a new product offering. By identifying and targeting these clients, the company can:

1. **Expand Market Opportunities**: Tap into clients who can benefit most from acquisition campaigns, thereby enhancing their performance.
2. **Boost Revenue**: Cross-sell acquisition campaigns to clients, providing them with an effective marketing strategy and increasing the company's revenue.
3. **Strengthen Client Relationships**: Offer tailored solutions that address specific client needs, improving satisfaction and loyalty.
""")

# Methodology Section
st.header("Methodology")
st.write("""
Our methodology can be broken down into the following steps:

1. **Collect and Clean Data**: Gather all relevant data needed for the analysis and perform data cleaning, feature engineering, handling outliers, etc. (these steps will not be shown in this demonstration).
2. **Data Exploration and Feature Selection**: Explore the data and select the most relevant features for the model.
3. **Train and Test AI Models**: Use the training data to develop AI models that can detect clients with potential based on annotated historical data.
4. **Evaluate and Select Best Model**: Evaluate the performance of different models and select the best one.
5. **Predict on New Data**: For clients without acquisition campaigns, use the selected model to predict which ones would benefit and output a list.
6. **A/B Testing**: Compare the performance of the output list against the previous methodology (randomly choosing clients).
7. **Evaluate Results**: Analyze the outcomes to determine the effectiveness of the new model.
""")

# Disclaimer Section
st.header("Disclaimer")
st.write("""
For simplicity, the dataset and steps shown have been cleaned, modified, and simplified to ensure privacy. 
The methodologies and results presented are based on these modified datasets.
""")

# Ending note
st.write("""
We hope you find the Breaking Model project insightful and beneficial. If you have any questions or feedback, feel free to reach out!
""")
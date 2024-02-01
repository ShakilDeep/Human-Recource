import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

# Placeholder for additional feature generation
def generate_additional_features(data, total_features=120):
    # Assuming the model expects a certain number of features but the input data has less
    # We will generate dummy features to match the model's expectation
    current_features = data.shape[1]
    additional_features_needed = total_features - current_features
    for i in range(additional_features_needed):  # Generate additional features as needed
        data[f'dummy_feature_{i}'] = 0  # Initialize additional features with 0
    return data

# Load the model
with open('human.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Create a function to preprocess the input data
def preprocess_input(data):
    # Convert 'Age' to numerical format
    age_mapping = {'<35': 0, '>35': 1}
    data['Age'] = data['Age'].map(age_mapping)
    
    # Convert 'EdLevel' to numerical format
    edlevel_mapping = {'NoHigherEd': 0, 'Other': 1, 'Undergraduate': 2, 'Master': 3, 'PhD': 4}
    data['EdLevel'] = data['EdLevel'].map(edlevel_mapping)
    
    # Encode 'Country' using LabelEncoder
    le = LabelEncoder()
    data['Country'] = le.fit_transform(data['Country'])
    
    # Process 'HaveWorkedWith' - split by commas, then binarize
    mlb = MultiLabelBinarizer()
    data = data.join(pd.DataFrame(mlb.fit_transform(data.pop('HaveWorkedWith').str.split(',')),
                                  columns=mlb.classes_,
                                  index=data.index))
    
    # Ensure all expected features are present, even if they're all zeros
    # Adjusted to handle models without 'feature_names_in_' attribute and KeyError exception
    try:
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
        else:
            expected_features = pd.DataFrame(columns=model.get_params()['features'])
    except KeyError:
        # If 'features' key is not found, default to an empty list
        expected_features = []
    
    missing_features = set(expected_features) - set(data.columns)
    for feature in missing_features:
        data[feature] = 0
    
    # Reorder columns to match the model's expectations, if expected_features is not empty
    if expected_features:
        data = data.reindex(columns=expected_features, fill_value=0)
    
    return data

# Create the Streamlit app
st.title('Employee Hiring Prediction')
st.write('Enter the employee details and click "Predict" to see if they will be hired.')

# Create the input fields
age = st.selectbox('Age', ['<35', '>35'])
edlevel = st.selectbox('Education Level', ['Master', 'Undergraduate', 'PhD', 'Other', 'NoHigherEd'])
yearscodepro = st.number_input('Years of Professional Coding Experience', min_value=0)
country = st.text_input('Country')
haveworkedwith = st.text_input('Technologies the Employee has Worked With (separated by commas)')

# Create the predict button
if st.button('Predict'):
    input_data = pd.DataFrame({'Age': [age], 'EdLevel': [edlevel], 'YearsCodePro': [yearscodepro], 'Country': [country], 'HaveWorkedWith': [haveworkedwith]})
    input_data = preprocess_input(input_data)
    # Ensure input_data has the correct number of features expected by the model
    if input_data.shape[1] < 120:
        input_data = generate_additional_features(input_data)  # Add dummy features if necessary
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.write('This employee will be hired.')
    else:
        st.write('This employee will not be hired.')


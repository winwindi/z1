import streamlit as st
import pandas as pd
import pickle
import datetime




# Load model
def load_model(model_name):
   if model_name == 'Random Forrest':
        model = pickle.load(open('models/no_resampling_rf_model.pkl', 'rb'))
   elif model_name == 'Logistic Regression':
        model = pickle.load(open('models/no_resampling_lr_model.pkl', 'rb'))
   elif model_name == 'SVM':
        model = pickle.load(open('models/no_resampling_svm_model.pkl', 'rb'))
   elif model_name == 'Smote_LR':
         model = pickle.load(open('models/smote_lr_model.pkl', 'rb'))
   elif model_name == 'Random Forest':
         model = pickle.load(open('models/random_forest_model.pkl', 'rb'))
   elif model_name == 'XGB':
         model = pickle.load(open('models/xgboost_model.pkl', 'rb'))
   elif model_name == 'GBM':
         model = pickle.load(open('models/gradient_boosting_model.pkl', 'rb'))
   return model



# Fungsi untuk melakukan prediksi
def predict_status(model, data):
    predictions = model.predict(data)
    return predictions

# Fungsi untuk mewarnai prediksi
def warna(wrn):
    color = 'pink' if wrn == 'Dropout' else 'blue'
    return f'color: {color}'

st.title('Jaya Jaya Institute Student Prediction using Machine Learning')

def main():
    st.write("by B244044F")       




tab1, tab2, tab3 = st.tabs([
    "Prediction",
    "tab2",
    "tab3"
])

with tab1:
    st.header("Student Prediction")

    with st.expander("How to run the prediction:"):
        st.write(
            """
                1. Choose machine learning model
                2. Upload Filetest.csv
                3. Click predict button
                4. Result will appear and can be 'Download (.csv)'. 
            """
        )


    
    model_name = st.radio("Choose Machine Learning Model", ('Smote_LR','Random Forrest',"Logistic Regression", "SVM"))


    # Upload File
    upload = st.file_uploader("Upload Filetest", type=["csv"])

    if upload is not None:
        data = pd.read_csv(upload)

        ID = data['ID']
        Name = data['Name']
        data = data.drop(columns=['ID', 'Name'])

        # Load model
        model = load_model(model_name)

        # click button
        if st.button('✨Predict'):
            
            predictions = predict_status(model, data)
        
            prediction_labels = ['Graduate' if pred == 1 else 'Dropout' for pred in predictions]

            # Result
            hasil = pd.DataFrame({
                'ID': ID,
                'Name': Name,
                'Status Prediction': prediction_labels
            })
         
            st.write("Prediction result:")
            st.dataframe(hasil.style.applymap(warna, subset=['Status Prediction']))

            # Download result
            csv = hasil.to_csv(index=False)
            st.download_button(
                label="Download Prediction Result",
                data=csv,
                file_name='Prediction result.csv',
                mime='text/csv'
            )

if __name__ == '__main__':
    main()



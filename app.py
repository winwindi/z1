import streamlit as st
import pandas as pd
import pickle
import datetime




# Load model
def load_model(model_name):
   if model_name == 'Random Forrest':
        model = pickle.load(open('model15/no_resampling_rf_model.pkl', 'rb'))
   elif model_name == 'Logistic Regression':
        model = pickle.load(open('model15/no_resampling_lr_model.pkl', 'rb'))
   elif model_name == 'SVM':
        model = pickle.load(open('model15/no_resampling_svm_model.pkl', 'rb'))
   elif model_name == 'Smote_LR':
         model = pickle.load(open('model15/smote_lr_model.pkl', 'rb'))
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

if __name__ == '__main__':
    main()



tab1, tab2, tab3 = st.tabs([
    "Prediction",
    "Dashboard",
    "About"
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


with tab2:
    st.subheader("Student dashboard")
    st.markdown("[View Dashboard on Tableau Public](https://public.tableau.com/app/profile/win.wikind/viz/studentdashboard_17394505558280/Dashboard1?publish=yes)")

    st.image("https://github.com/user-attachments/assets/e808d9fa-6c30-45fb-9d21-a5161f42cf98")


    with st.expander("Summary: "):
        st.write(
            """
            1. High Dropout Rate (32.12%)

            Main contributing factors: academic debt, low admission scores, and delayed tuition payments.
            Students with scholarships have a higher graduation rate compared to those without scholarships.
            
            2. Large Number of Displaced Students (54.84%)

            Additional support, both financial and academic, is needed to improve their retention.
            
            3. Financial Issues Affect Academic Status

            Students with academic debt are more vulnerable to dropping out compared to those without debt.
            
            4. Some Study Programs Have High Dropout Rates

            Nursing and Management programs have a significant dropout rate.
            Study programs with a small number of students need to be evaluated to enhance their appeal.
            
            """
        )


    with st.expander("Recommendation: "):
            st.write(
                """
                Proposed Strategies to Reduce Dropout Rates

                1. Expansion of Scholarship Programs

                    Focus on students at risk of dropping out based on their financial status.
                2. Flexible Payment Options

                    Provide installment payment plans with 0% interest or partial debt/tuition forgiveness programs for high-achieving students.
                3. Gap Year Program with Academic Support

                    Offer a structured gap year program with special orientation for returning students to help them adjust to changes in the environment, technology, or curriculum.
                4. Academic Guidance for Students with Low Grades

                    Additional learning programs for students with low admission scores, especially for older and married students.
                5. Psychological & Counseling Support

                    Establish a psychological support and academic counseling center for students experiencing financial or academic stress.
                6. Monitoring and Predicting Dropouts

                    Conduct further review and monitoring of currently enrolled students.
                    Utilize machine learning technology to predict dropout risks, enabling early intervention.
                
                """
            )

    st.write("Based on this analysis, the university must strengthen its financial, academic, and psychological support programs to improve student success. By implementing more flexible and data-driven strategies, dropout rates can be significantly reduced, and the quality of education at Jaya Jaya Institute will continue to improve."
            )
    

with tab3:
    
    st.write(
            """
            
            Jaya Jaya Institute is a higher education institution that has been established since the year 2000. Over the years, it has produced many graduates with an excellent reputation. However, a significant number of students fail to complete their education and drop out.
            The high dropout rate is a major concern for an educational institution. Therefore, Jaya Jaya Institute aims to detect students at risk of dropping out as early as possible so that they can receive specialized guidance.

            """
            )

with st.sidebar:
        st.subheader("Disclaimer")
        st.markdown("This document is created for academic purposes and is not intended to represent any specific individuals or organizations. Any similarities in names or events are purely coincidental.")


year_now = datetime.date.today().year
year = year_now if year_now == 2024 else f'2024 - {year_now}'
name = "[B244044F]"
copyright = 'Copyright © ' + str(year) + ' ' + name
st.caption(copyright)

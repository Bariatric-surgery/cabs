# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 12:45:38 2023

"""

# Load libraries
import streamlit as st
import scipy.stats as sta
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import numpy as np
plt.style.use('grayscale')

# Load the model
model_single = pkl.load(open(r'models/MLP_Optimized.sav' , 'rb'))
model_multioutput = pkl.load(open(r'models/MLP_Optimized_Option_1.sav' , 'rb'))

# Load the data to calculate confident interval
path = r'data'
name = '/CABS_V202301.csv'
data_types = ['diabetes_mellitus_after_surgery' ,
              'osas_after_surgery']
data = pd.read_csv(path  + name , sep = ',')
for col in data_types:
  data[col] = data[col].fillna(-1)
  data[col] = data[col].astype(int)
  data[col] = data[col].astype(str)
  data[col] = data[col].replace('-1', np.nan)
bmi_columns = ['bmi_pre' , 'bmi_6_months','bmi_12_months','bmi_18_months','bmi_2years', 'bmi_3years', 'bmi_4years', 'bmi_5years']
data = data[bmi_columns]

# Calculate confidence interval of each bmi value
confident_interval = {}
for i in data.columns.tolist():
  confident_interval[i] = sta.t.interval(alpha = 0.95,
                                        df = data[i].dropna().shape[0],
                                        loc = np.mean(data[i].dropna().values),
                                        scale = sta.sem(data[i].dropna().values))
# Calculate center value
mean_bmi = data.mean(axis = 0).values.tolist()
upper = [confident_interval[i][1] for i in confident_interval.keys()]
lower = [confident_interval[i][0] for i in confident_interval.keys()]
# Calculate de diferences
diferences_down = [lower[i] - mean_bmi[i] for i in range(len(mean_bmi))]
diferences_up = [upper[i] - mean_bmi[i] for i in range(len(mean_bmi))]


# Page configuration
st.set_page_config(
    page_title="BMI Prediction App"
)
st.set_option('deprecation.showPyplotGlobalUse', False)

######################
#main page layout
######################

st.title("BMI Prediction")
st.subheader("Description")

st.subheader("To predict BMI value, you need to follow the steps below:")
st.markdown("""
1. Enter clinical parameters of patient on the left side bar.
2. Press the "Predict" button and wait for the result.
""")

st.subheader("Prediction result: ")

######################
#sidebar layout
######################

st.sidebar.title("Patiens Info")
st.sidebar.write("Please choose parameters")

# input features
sex = st.sidebar.selectbox('Please choose gender', ('Male' , 'Female'))
age_years = st.sidebar.slider("Please choose age:", min_value = 18, max_value = 100,step = 1)
asa = st.sidebar.selectbox('Please choose asa score', ('Healthy Person', 'Mid systemic disease', 'Severe systemic disease', 'Severe systemic disease that is a constant threat to life'))
charleson_comorbidity_index = st.sidebar.selectbox('Please choose Charleson Comorbidity index', (0, 1, 2, 3, 4, 5, 6 , 7))
depression = st.sidebar.selectbox('Please choose if the patient has depression', ('Yes', 'No', 'Unknown'))
prior_abdominal_surgery = st.sidebar.selectbox('Please choose prior abdominal surgery type', ('Yes', 'No', 'Unknown'))
diabetes_mellitus_preoperative = st.sidebar.selectbox('Please choose if the patient has diabetes mellitus preoperative', ('Yes' , 'No'))
diabetes_mellitus_after_surgery = st.sidebar.selectbox('Please choose if the patient has diabetes mellitus after surgery', ('Yes' , 'No'))
antidiabetic_drugs_preoperativ___1 = st.sidebar.selectbox('Please choose if the patient has Orale Antidiabetic drugs', ('Yes' , 'No'))
antidiabetic_drugs_preoperativ___2 = st.sidebar.selectbox('Please choose if the patient has Long-Acting Insulin', ('Yes' , 'No'))
antidiabetic_drugs_preoperativ___3 = st.sidebar.selectbox('Please choose if the patient has Intermediate-Acting Insulin', ('Yes' , 'No'))
antidiabetic_drugs_preoperativ___4 = st.sidebar.selectbox('Please choose if the patient has Short-Acting insulin', ('Yes' , 'No'))
postoperative_hba1c_6_months = st.sidebar.number_input("Please choose postoperative hba1c after 6 month of the surgery:")
postoperative_hba1c_12_months = st.sidebar.number_input("Please choose postoperative hba1c after 12 month of the surgery:")
osas_preoperaative = st.sidebar.selectbox('Please choose if the patient has Obstructive Sleep Apnea Syndrome (OSAS) preoperative', ('Yes' , 'No'))
osas_after_surgery = st.sidebar.selectbox('Please choose if the patient has Obstructive Sleep Apnea Snydrome (OSAS) after surgery', ('Yes' , 'No'))
surgery = st.sidebar.selectbox('Please choose surgery type', ('Laparoscopiy Sleeve Gastrectomy', 'Roux en-Y Gastric Bypass'))
conversion_f = st.sidebar.selectbox('Please choose conversion from gastric sleeve to gastric bypass', ('Yes', 'No', 'Unknown'))
surgery_time = st.sidebar.slider("Please choose surgery time (min):", min_value = 25, max_value = 352,step = 1)
preventive_closure_of_mesenteric_defects = st.sidebar.selectbox('Please choose cpreventive_closure_of_mesenteric_defects type', (1, 2, 3))
complication___1 = st.sidebar.selectbox('Please choose if the patient has Anastomotic leakage', ('Yes' , 'No'))
complication___2 = st.sidebar.selectbox('Please choose if the patient has Gastric leakage', ('Yes' , 'No'))
complication___3 = st.sidebar.selectbox('Please choose if the patient has Intussusception after Roux-en-Y gastric bypass', ('Yes' , 'No'))
complication___4 = st.sidebar.selectbox('Please choose if the patient has Mesenteric internal hernia after Roux-en-Y gastric bypass', ('Yes' , 'No'))
complication___5 = st.sidebar.selectbox('Please choose if the patient has Internal hernia through Peterson´s defect after Roux-en-Y gastric bypass', ('Yes' , 'No'))
complication___6 = st.sidebar.selectbox('Please choose if the patient has Hiatal hernia', ('Yes' , 'No'))
complication___7 = st.sidebar.selectbox('Please choose if the patient has Gastro Esophageal Refux Disease (GERD)', ('Yes' , 'No'))
complication___8 = st.sidebar.selectbox('Please choose if the patient has No Complication', ('Yes' , 'No'))
complication___9 = st.sidebar.selectbox('Please choose if the patient has Amastomotic Ulcera', ('Yes' , 'No'))
the_clavien_dindo_classification = st.sidebar.selectbox('Please choose the_clavien_dindo_classification', ('I: Any deviation from the normal postoperative course without the need for pharmacological treatment or surgical',
                                                                                                           'II: Requiring pharmacological treatment with drugs other than such allowed for grade I complications',
                                                                                                           'IIIa: Intervention not under general anesthesia',
                                                                                                           'IIIb: Intervention under general anesthesia',
                                                                                                           'IVa: single organ dysfunction (including dialysis)',
                                                                                                           'Vb: multiorgandysfunction',
                                                                                                           'V Death of a patient',
                                                                                                           'No complication'))
bmi_pre = st.sidebar.number_input("Please enter BMI pre value:")
bmi_t = st.sidebar.number_input("Please enter BMI(t) value, if you are going to predict BMI 6 monhts, puth here BMI pre value again:")

# Function that build the X input for the model
def preprocess_input(sex , age_years , asa , charleson_comorbidity_index, depression,
                     prior_abdominal_surgery, diabetes_mellitus_preoperative, diabetes_mellitus_after_surgery,
                     antidiabetic_drugs_preoperativ___1 , antidiabetic_drugs_preoperativ___2, antidiabetic_drugs_preoperativ___3 , antidiabetic_drugs_preoperativ___4,
                     postoperative_hba1c_6_months , postoperative_hba1c_12_months , osas_preoperaative, osas_after_surgery,
                     surgery, conversion_f, surgery_time, preventive_closure_of_mesenteric_defects, complication___1,
                     complication___2, complication___3, complication___4, complication___5, complication___6,
                     complication___7, complication___8, complication___9, the_clavien_dindo_classification, bmi_t , bmi_pre):
    
    information = {}
    
    # Parser sex
    if sex == 'Male':
        information['sex'] = [1]
    if sex == 'Female':
        information['sex'] = [2]
    # Parser age
    information['age_years'] = [int(age_years)]
    # Parser asa
    if asa == 'Healthy Person':
        information['asa'] = [1]
    if asa == 'Mid systemic disease':
        information['asa'] = [2]
    if asa == 'Severe systemic disease':
        information['asa'] = [3]
    if asa == 'Severe systemic disease that is a constant threat to life':
        information['asa'] = [4]
    # Parser charleson_comorbidity_index
    information['charleson_comorbidity_index'] = [int(charleson_comorbidity_index)]
    # Parser depression
    if depression == 'Yes':
        information['depression'] = [1]
    if depression == 'No':
        information['depression'] = [2]
    if depression == 'Unknown':
        information['depression'] = [3]
    # Parser prior_abdominal_surgery
    if prior_abdominal_surgery == 'Yes':
        information['prior_abdominal_surgery'] = [1]
    if prior_abdominal_surgery == 'No':
        information['prior_abdominal_surgery'] = [2]
    if prior_abdominal_surgery == 'Unknown':
        information['prior_abdominal_surgery'] = [3]
    # Parser diabetes_mellitus_preoperative
    if diabetes_mellitus_preoperative == 'Yes':
        information['diabetes_mellitus_preoperative'] = [1]
    if diabetes_mellitus_preoperative == 'No':
        information['diabetes_mellitus_preoperative'] = [0]
    # Parser diabetes_mellitus_after_surgery
    if diabetes_mellitus_after_surgery == 'Yes':
        information['diabetes_mellitus_after_surgery'] = [1]
    if diabetes_mellitus_after_surgery == 'No':
        information['diabetes_mellitus_after_surgery'] = [0]
    # Parser antidiabetic_drugs_preoperativ___1
    if antidiabetic_drugs_preoperativ___1 == 'Yes':
        information['antidiabetic_drugs_preoperativ___1'] = [1]
    if antidiabetic_drugs_preoperativ___1 == 'No':
        information['antidiabetic_drugs_preoperativ___1'] = [0]
    # Parser antidiabetic_drugs_preoperativ___2
    if antidiabetic_drugs_preoperativ___2 == 'Yes':
        information['antidiabetic_drugs_preoperativ___2'] = [1]
    if antidiabetic_drugs_preoperativ___2 == 'No':
        information['antidiabetic_drugs_preoperativ___2'] = [0]
    # Parser antidiabetic_drugs_preoperativ___3
    if antidiabetic_drugs_preoperativ___3 == 'Yes':
        information['antidiabetic_drugs_preoperativ___3'] = [1]
    if antidiabetic_drugs_preoperativ___3 == 'No':
        information['antidiabetic_drugs_preoperativ___3'] = [0]
    # Parser antidiabetic_drugs_preoperativ___4
    if antidiabetic_drugs_preoperativ___4 == 'Yes':
        information['antidiabetic_drugs_preoperativ___4'] = [1]
    if antidiabetic_drugs_preoperativ___4 == 'No':
        information['antidiabetic_drugs_preoperativ___4'] = [0]
    # Parser postoperative_hba1c_6_month
    information['postoperative_hba1c_6_months'] = [float(postoperative_hba1c_6_months)]
    # Parser postoperative_hba1c_12_month
    information['postoperative_hba1c_12_months'] = [float(postoperative_hba1c_12_months)]
    # Parser osas_preoperaative
    if osas_preoperaative == 'Yes':
        information['osas_preoperaative'] = [1]
    if osas_preoperaative == 'No':
        information['osas_preoperaative'] = [0]
    # Parser osas_after_surgery
    if osas_after_surgery == 'Yes':
        information['osas_after_surgery'] = [1]
    if osas_after_surgery == 'No':
        information['osas_after_surgery'] = [0]
    # Parser surgery
    if surgery == 'Laparoscopiy Sleeve Gastrectomy':
        information['surgery'] = [1]
    if surgery == 'Roux en-Y Gastric Bypass':
        information['surgery'] = [2]
    # Parser conversion_f
    if conversion_f == 'Yes':
        information['conversion_f'] = [1]
    if conversion_f == 'No':
        information['conversion_f'] = [2]
    if conversion_f == 'Unknown':
        information['conversion_f'] = [3]
    # Parser surgery_time
    information['surgery_time'] = [int(surgery_time)]
    # Parser preventive_closure_of_mesenteric_defects
    information['preventive_closure_of_mesenteric_defects'] = [int(preventive_closure_of_mesenteric_defects)]
    # Parser complication___1
    if complication___1 == 'Yes':
        information['complication___1'] = [1]
    if complication___1 == 'No':
        information['complication___1'] = [0]
    # Parser complication___2
    if complication___2 == 'Yes':
        information['complication___2'] = [1]
    if complication___2 == 'No':
        information['complication___2'] = [0]
    # Parser complication___3
    if complication___3 == 'Yes':
        information['complication___3'] = [1]
    if complication___3 == 'No':
        information['complication___3'] = [0]
    # Parser complication___4
    if complication___4 == 'Yes':
        information['complication___4'] = [1]
    if complication___4 == 'No':
        information['complication___4'] = [0]
    # Parser complication___5
    if complication___5 == 'Yes':
        information['complication___5'] = [1]
    if complication___5 == 'No':
        information['complication___5'] = [0]
    # Parser complication___6
    if complication___6 == 'Yes':
        information['complication___6'] = [1]
    if complication___6 == 'No':
        information['complication___6'] = [0]
    # Parser complication___6
    if complication___7 == 'Yes':
        information['complication___7'] = [1]
    if complication___7 == 'No':
        information['complication___7'] = [0]
    # Parser complication___8
    if complication___8 == 'Yes':
        information['complication___8'] = [1]
    if complication___8 == 'No':
        information['complication___8'] = [0]
    # Parser complication___9
    if complication___9 == 'Yes':
        information['complication___9'] = [1]
    if complication___9 == 'No':
        information['complication___9'] = [0]
    # Parser the_clavien_dindo_classification
    if the_clavien_dindo_classification == 'I: Any deviation from the normal postoperative course without the need for pharmacological treatment or surgical':
        information['the_clavien_dindo_classification'] = [1]
    if the_clavien_dindo_classification == 'II: Requiring pharmacological treatment with drugs other than such allowed for grade I complications':
        information['the_clavien_dindo_classification'] = [2]
    if the_clavien_dindo_classification == 'IIIa: Intervention not under general anesthesia':
        information['the_clavien_dindo_classification'] = [3]
    if the_clavien_dindo_classification == 'IIIb: Intervention under general anesthesia':
        information['the_clavien_dindo_classification'] = [4]
    if the_clavien_dindo_classification == 'IVa: single organ dysfunction (including dialysis)':
        information['the_clavien_dindo_classification'] = [5]
    if the_clavien_dindo_classification == 'Vb: multiorgandysfunction':
        information['the_clavien_dindo_classification'] = [6]
    if the_clavien_dindo_classification == 'V Death of a patient':
        information['the_clavien_dindo_classification'] = [7]
    if the_clavien_dindo_classification == 'No complication':
        information['the_clavien_dindo_classification'] = [8]
    # Parser bmi_pre
    #information['bmi_pre'] = [float(bmi_pre)]
    # Parser bmi_t
    information['BMI(t)'] = [float(bmi_t)]
    # Parser bmi_pre
    information['bmi_pre'] = float(bmi_pre)
    
    # Convert to a dataframe
    information = pd.DataFrame(data = information)
    
    return information

# Parser user information
user_input = preprocess_input(sex, age_years, asa, charleson_comorbidity_index, depression, prior_abdominal_surgery, diabetes_mellitus_preoperative, diabetes_mellitus_after_surgery, antidiabetic_drugs_preoperativ___1, antidiabetic_drugs_preoperativ___2, antidiabetic_drugs_preoperativ___3, antidiabetic_drugs_preoperativ___4, postoperative_hba1c_6_months, postoperative_hba1c_12_months, osas_preoperaative, osas_after_surgery, surgery, conversion_f, surgery_time, preventive_closure_of_mesenteric_defects, complication___1, complication___2, complication___3, complication___4, complication___5, complication___6, complication___7, complication___8, complication___9, the_clavien_dindo_classification, bmi_t , bmi_pre)

# Prediction
predict_button = st.button('Predict')
if predict_button:
    # Predict the single BMI(t+1) value
    pred = model_single.predict(user_input.drop(columns = ['bmi_pre']))
    st.header("BMI(t+1):")
    st.subheader(str(round(pred[0] , 2)) + ' (' + str(round(pred[0] + np.mean(diferences_down) , 2)) + ' ; ' + str(round(pred[0] + np.mean(diferences_up) , 2)) + ')')
    # Predict all BMI values at once using patients information and bmi pre
    bmi_columns = ['bmi_6_months','bmi_12_months','bmi_18_months','bmi_2years', 'bmi_3years', 'bmi_4years', 'bmi_5years']
    pred_multi = pd.DataFrame(model_multioutput.predict(user_input.drop(columns = ['BMI(t)'])) , columns = bmi_columns).T
    pred_multi.columns = ['BMI Value']
    pred_multi['Lower Diference'] = pd.DataFrame(diferences_down[1:] , index = pred_multi.index.tolist())
    pred_multi['Upper Diference'] = pd.DataFrame(diferences_up[1:] , index = pred_multi.index.tolist())
    pred_multi['Lower Bound'] = pred_multi['BMI Value'] + pred_multi['Lower Diference']
    pred_multi['Upper Bound'] = pred_multi['BMI Value'] + pred_multi['Upper Diference']
    pred_multi = pred_multi.drop(columns = ['Lower Diference' , 'Upper Diference'])
    # Plot the bmi values and their confident interval
    x = ['6M' , '12M' , '18M' , '2Y' , '3Y' , '4Y' , '5Y']
    # Create the plot
    st.header("Curve Plot of Evolution of BMI")
    y = pred_multi['BMI Value'].values.tolist()
    fig, ax = plt.subplots()
    x = ['6M' , '12M' , '18M' , '2Y' , '3Y' , '4Y' , '5Y']
    ax.plot(x, y)
    ax.fill_between(
        x, pred_multi['Lower Bound'], pred_multi['Upper Bound'], color='gray', alpha=.15)
    ax.set_title('Evolution of BMI Value')
    ax.set_xlabel('BMI Time Step', 
               fontweight ='bold')
    ax.set_ylabel('BMI',fontweight ='bold')
    fig.autofmt_xdate(rotation=45)
    st.pyplot(fig)
    # Show the table with the future BMIs
    st.header("Future BMIs:")
    for i in pred_multi.columns.tolist():
        pred_multi[i] = pred_multi[i].apply(lambda y : "{0:.2f}".format(y))
    st.dataframe(pred_multi)

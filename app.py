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
from streamlit_option_menu import option_menu
import numpy as np
plt.style.use('grayscale')

###############################################################################
# Section when the app initialize and load the required information
@st.cache_data() # We use this decorator so this initialization doesn't run every time the user change into the page
def initialize_app():
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

    return model_single, model_multioutput, data, bmi_columns, diferences_down, diferences_up
###############################################################################
# Section with the functions that the app will use
# Function that build the X input for the model
def preprocess_input_case_2(sex , age_years , asa , charleson_comorbidity_index, depression,
                            prior_abdominal_surgery, diabetes_mellitus_preoperative, diabetes_mellitus_after_surgery,
                            antidiabetic_drugs_preoperativ___1 , antidiabetic_drugs_preoperativ___2, antidiabetic_drugs_preoperativ___3 , antidiabetic_drugs_preoperativ___4,
                            postoperative_hba1c_6_months , postoperative_hba1c_12_months , osas_preoperaative, osas_after_surgery,
                            surgery, conversion_f, surgery_time, preventive_closure_of_mesenteric_defects, complication___1,
                            complication___2, complication___3, complication___4, complication___5, complication___6,
                            complication___7, complication___8, complication___9, the_clavien_dindo_classification, bmi_pre, bmi_6_months,
                            bmi_12_months, bmi_18_months, bmi_2_years, bmi_3_years, bmi_4_years, bmi_5_years):
    
    information = {}
    
    # Parser sex
    if sex == 'Male':
        information['sex'] = [1]
    if sex == 'Female':
        information['sex'] = [2]
    # Parser age
    information['age_years'] = [int(age_years)]
    # Parser asa
    if asa == 'I':
        information['asa'] = [1]
    if asa == 'II':
        information['asa'] = [2]
    if asa == 'III':
        information['asa'] = [3]
    if asa == 'IV':
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
    if surgery == 'Laparoscopy Sleeve Gastrectomy':
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
    if the_clavien_dindo_classification == 'I':
        information['the_clavien_dindo_classification'] = [1]
    if the_clavien_dindo_classification == 'II':
        information['the_clavien_dindo_classification'] = [2]
    if the_clavien_dindo_classification == 'IIIa':
        information['the_clavien_dindo_classification'] = [3]
    if the_clavien_dindo_classification == 'IIIb':
        information['the_clavien_dindo_classification'] = [4]
    if the_clavien_dindo_classification == 'IVa':
        information['the_clavien_dindo_classification'] = [5]
    if the_clavien_dindo_classification == 'IVb':
        information['the_clavien_dindo_classification'] = [6]
    if the_clavien_dindo_classification == 'V':
        information['the_clavien_dindo_classification'] = [7]
    if the_clavien_dindo_classification == 'No complication':
        information['the_clavien_dindo_classification'] = [8]
    # Parser bmi_pre
    information['bmi_pre'] = [float(bmi_pre)]
    # Parser bmi_6_months
    information['bmi_6_months'] = [float(bmi_6_months)]
    # Parser bmi_12_months
    information['bmi_12_months'] = [float(bmi_12_months)]
    # Parser bmi_18_months
    information['bmi_18_months'] = [float(bmi_18_months)]
    # Parser bmi_2_years
    information['bmi_2years'] = [float(bmi_2_years)]
    # Parser bmi_3_years
    information['bmi_3years'] = [float(bmi_3_years)]
    # Parser bmi_4_years
    information['bmi_4years'] = [float(bmi_4_years)]
    # Parser bmi_5_years
    information['bmi_5years'] = [float(bmi_5_years)]
    # Parser bmi_t
    information['BMI(t)'] = [-1]
    if information['bmi_6_months'] != [-1]:
        information['BMI(t)'] = [float(bmi_6_months)]
    if information['bmi_12_months'] != [-1]:
        information['BMI(t)'] = [float(bmi_12_months)]
    if information['bmi_18_months'] != [-1]:
        information['BMI(t)'] = [float(bmi_18_months)]
    if information['bmi_2years'] != [-1]:
        information['BMI(t)'] = [float(bmi_2_years)]
    if information['bmi_3years'] != [-1]:
        information['BMI(t)'] = [float(bmi_3_years)]
    if information['bmi_4years'] != [-1]:
        information['BMI(t)'] = [float(bmi_4_years)]
    if information['bmi_5years'] != [-1]:
        information['BMI(t)'] = [float(bmi_5_years)]
    if information['BMI(t)'] == [-1]:
        information['BMI(t)'] = [float(bmi_pre)]
    # Convert to a dataframe
    information = pd.DataFrame(data = information)
    
    return information

# Function that build the X input for the model
def preprocess_input_case_1(sex , age_years , asa , charleson_comorbidity_index, depression,
                     prior_abdominal_surgery, diabetes_mellitus_preoperative, diabetes_mellitus_after_surgery,
                     antidiabetic_drugs_preoperativ___1 , antidiabetic_drugs_preoperativ___2, antidiabetic_drugs_preoperativ___3 , antidiabetic_drugs_preoperativ___4,
                     postoperative_hba1c_6_months , postoperative_hba1c_12_months , osas_preoperaative, osas_after_surgery,
                     surgery, conversion_f, surgery_time, preventive_closure_of_mesenteric_defects, complication___1,
                     complication___2, complication___3, complication___4, complication___5, complication___6,
                     complication___7, complication___8, complication___9, the_clavien_dindo_classification, bmi_pre):
    
    information = {}
    
    # Parser sex
    if sex == 'Male':
        information['sex'] = [1]
    if sex == 'Female':
        information['sex'] = [2]
    # Parser age
    information['age_years'] = [int(age_years)]
    # Parser asa
    if asa == 'I':
        information['asa'] = [1]
    if asa == 'II':
        information['asa'] = [2]
    if asa == 'III':
        information['asa'] = [3]
    if asa == 'IV':
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
    if surgery == 'Laparoscopy Sleeve Gastrectomy':
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
    if the_clavien_dindo_classification == 'I':
        information['the_clavien_dindo_classification'] = [1]
    if the_clavien_dindo_classification == 'II':
        information['the_clavien_dindo_classification'] = [2]
    if the_clavien_dindo_classification == 'IIIa':
        information['the_clavien_dindo_classification'] = [3]
    if the_clavien_dindo_classification == 'IIIb':
        information['the_clavien_dindo_classification'] = [4]
    if the_clavien_dindo_classification == 'IVa':
        information['the_clavien_dindo_classification'] = [5]
    if the_clavien_dindo_classification == 'IVb':
        information['the_clavien_dindo_classification'] = [6]
    if the_clavien_dindo_classification == 'V':
        information['the_clavien_dindo_classification'] = [7]
    if the_clavien_dindo_classification == 'No complication':
        information['the_clavien_dindo_classification'] = [8]
    # Parser bmi_pre
    information['bmi_pre'] = [float(bmi_pre)]
    # Parser bmi_t
    information['BMI(t)'] = [float(bmi_pre)]
    # Convert to a dataframe
    information = pd.DataFrame(data = information)
    
    return information
###############################################################################

# Page configuration
st.set_page_config(
    page_title="BMI Prediction App"
)
st.set_option('deprecation.showPyplotGlobalUse', False)
# Initialize app
model_single, model_multioutput, data, bmi_columns, diferences_down, diferences_up = initialize_app()

# Option Menu configuration
selected = option_menu(
    menu_title = 'Main Menu',
    options = ['Home' , 'Case 1' , 'Case 2'],
    icons = ['house' , 'book' , 'evenlope'],
    menu_icon = 'cast',
    default_index = 0,
    orientation = 'horizontal')

######################
# Home page layout
######################
if selected == 'Home':
    st.title('BMI Prediction App')
    st.markdown("""
    This app contains 3 sections which you can access from the horizontal menu above.\n
    The sections are:\n
    Home: The main page of the app.\n
    Case 1: On this section you can predict the BMI future values of the patient when you don't have information for future BMI\n
    Case 2: On this section you can predict future BMI values based on the real BMI values you already have.
    """)
####################
# Case 1 page layout
####################
if selected == 'Case 1':
    st.title("BMI Prediction")
    st.subheader("Description")

    st.subheader("To predict BMI value, you need to follow the steps below:")
    st.markdown("""
    1. Enter clinical parameters of patient on the left side bar.
    2. Press the "Predict" button and wait for the result.
    """)
    

    #sidebar layout
    
    st.sidebar.title("Patiens Info")
    st.sidebar.subheader("Please choose parameters")
    
    # input features
    sex = st.sidebar.selectbox('Gender', ('Male' , 'Female'))
    age_years = st.sidebar.slider("Age:", min_value = 18, max_value = 100,step = 1)
    asa = st.sidebar.selectbox('ASA Score', ('I', 'II', 'III', 'IV'))
    charleson_comorbidity_index = st.sidebar.selectbox('Charleson Comorbidity index', (0, 1, 2, 3, 4, 5, 6 , 7))
    depression = st.sidebar.selectbox('Depression', ('Yes', 'No', 'Unknown'))
    prior_abdominal_surgery = st.sidebar.selectbox('Prior Abdominal surgery type', ('Yes', 'No', 'Unknown'))
    diabetes_mellitus_preoperative = st.sidebar.selectbox('Diabetes Mellitus preoperative', ('Yes' , 'No'))
    diabetes_mellitus_after_surgery = st.sidebar.selectbox('Diabetes Mellitus after surgery', ('Yes' , 'No'))
    antidiabetic_drugs_preoperativ___1 = st.sidebar.selectbox('Orale Antidiabetic drugs', ('Yes' , 'No'))
    antidiabetic_drugs_preoperativ___2 = st.sidebar.selectbox('Long-Acting Insulin', ('Yes' , 'No'))
    antidiabetic_drugs_preoperativ___3 = st.sidebar.selectbox('Intermediate-Acting Insulin', ('Yes' , 'No'))
    antidiabetic_drugs_preoperativ___4 = st.sidebar.selectbox('Short-Acting Insulin', ('Yes' , 'No'))
    postoperative_hba1c_6_months = st.sidebar.number_input("Postoperative hba1c after 6 month of the surgery:")
    postoperative_hba1c_12_months = st.sidebar.number_input("Postoperative hba1c after 12 month of the surgery:")
    osas_preoperaative = st.sidebar.selectbox('Obstructive Sleep Apnea Syndrome (OSAS) preoperative', ('Yes' , 'No'))
    osas_after_surgery = st.sidebar.selectbox('Obstructive Sleep Apnea Snydrome (OSAS) after surgery', ('Yes' , 'No'))
    surgery = st.sidebar.selectbox('Surgery type', ('Laparoscopy Sleeve Gastrectomy', 'Roux en-Y Gastric Bypass'))
    conversion_f = st.sidebar.selectbox('Conversion from gastric sleeve to gastric bypass', ('Yes', 'No', 'Unknown'))
    surgery_time = st.sidebar.slider("Surgery time (min):", min_value = 25, max_value = 352,step = 1)
    preventive_closure_of_mesenteric_defects = st.sidebar.selectbox('Preventive Closure of Mesenteric Defects type', (1, 2, 3))
    complication___1 = st.sidebar.selectbox('Anastomotic leakage', ('Yes' , 'No'))
    complication___2 = st.sidebar.selectbox('Gastric leakage', ('Yes' , 'No'))
    complication___3 = st.sidebar.selectbox('Intussusception after Roux-en-Y gastric bypass', ('Yes' , 'No'))
    complication___4 = st.sidebar.selectbox('Mesenteric internal hernia after Roux-en-Y gastric bypass', ('Yes' , 'No'))
    complication___5 = st.sidebar.selectbox('Internal hernia through Peterson´s defect after Roux-en-Y gastric bypass', ('Yes' , 'No'))
    complication___6 = st.sidebar.selectbox('Hiatal hernia', ('Yes' , 'No'))
    complication___7 = st.sidebar.selectbox('Gastro Esophageal Refux Disease (GERD)', ('Yes' , 'No'))
    complication___8 = st.sidebar.selectbox('No Complication', ('Yes' , 'No'))
    complication___9 = st.sidebar.selectbox('Anastomotic Ulcera', ('Yes' , 'No'))
    the_clavien_dindo_classification = st.sidebar.selectbox('Clavien-Dindo-Classification', ('I',
                                                                                             'II',
                                                                                             'IIIa',
                                                                                             'IIIb',
                                                                                             'IVa',
                                                                                             'IVb',
                                                                                             'V',
                                                                                             'No complication'))
    st.subheader("BMI Inputs: ")
    bmi_pre = st.number_input("BMI pre value:")
    st.subheader("Prediction result: ")
    # Parser user information
    user_input = preprocess_input_case_1(sex, age_years, asa, charleson_comorbidity_index, depression, prior_abdominal_surgery, diabetes_mellitus_preoperative, diabetes_mellitus_after_surgery, antidiabetic_drugs_preoperativ___1, antidiabetic_drugs_preoperativ___2, antidiabetic_drugs_preoperativ___3, antidiabetic_drugs_preoperativ___4, postoperative_hba1c_6_months, postoperative_hba1c_12_months, osas_preoperaative, osas_after_surgery, surgery, conversion_f, surgery_time, preventive_closure_of_mesenteric_defects, complication___1, complication___2, complication___3, complication___4, complication___5, complication___6, complication___7, complication___8, complication___9, the_clavien_dindo_classification, bmi_pre)
    # Prediction
    predict_button = st.button('Predict')
    if predict_button:
        # Check that bmi pre is equal or greater than 35
        number_of_warnings = 0
        if user_input['bmi_pre'].values[0] < 35:
            st.warning('The value of BMI 6 pre is lower than 35, check the input', icon="⚠️")
            number_of_warnings += 1
        if number_of_warnings == 0:
            #pred = model_single.predict(user_input.drop(columns = ['bmi_pre']))
            #st.header("BMI(t+1):")
            #st.subheader(str(round(pred[0] , 2)) + ' (' + str(round(pred[0] + np.mean(diferences_down) , 2)) + ' ; ' + str(round(pred[0] + np.mean(diferences_up) , 2)) + ')')
            # Predict all BMI values at once using patients information and bmi pre
            bmi_columns = ['bmi_6_months','bmi_12_months','bmi_18_months','bmi_2years', 'bmi_3years', 'bmi_4years', 'bmi_5years']
            pred_multi = pd.DataFrame(model_multioutput.predict(user_input.drop(columns = ['BMI(t)'])) , columns = bmi_columns).T
            pred_multi.columns = ['BMI Value']
            # Add bmi pre
            pred_multi = pd.concat([pd.DataFrame({'BMI Value' : [user_input['bmi_pre'].values[0]]}),
                                    pred_multi] , axis = 0)
            pred_multi['Lower Difference'] = pd.DataFrame(diferences_down[0:] , index = pred_multi.index.tolist())
            pred_multi['Upper Difference'] = pd.DataFrame(diferences_up[0:] , index = pred_multi.index.tolist())
            pred_multi['Lower Bound'] = pred_multi['BMI Value'] + pred_multi['Lower Difference']
            pred_multi['Upper Bound'] = pred_multi['BMI Value'] + pred_multi['Upper Difference']
            pred_multi = pred_multi.drop(columns = ['Lower Difference' , 'Upper Difference'])
            # Plot the bmi values and their confident interval
            x = ['PRE' ,'6M' , '12M' , '18M' , '2Y' , '3Y' , '4Y' , '5Y']
            # Create the plot
            st.header("Curve Plot of Evolution of BMI")
            y = pred_multi['BMI Value'].values.tolist()
            fig, ax = plt.subplots()
            x = ['PRE' , '6M' , '12M' , '18M' , '2Y' , '3Y' , '4Y' , '5Y']
            ax.plot(x[:2], y[:2] , color = 'blue' , label = 'Real BMI Values')
            ax.plot(x[1:], y[1:] , color = 'black' , label = 'Predicted BMI Values')
            ax.fill_between(
                x, pred_multi['Lower Bound'], pred_multi['Upper Bound'], color='gray', alpha=.15)
            ax.set_title('Evolution of BMI Value')
            ax.set_xlabel('BMI Time Step', 
                       fontweight ='bold')
            ax.set_ylabel('BMI',fontweight ='bold')
            fig.autofmt_xdate(rotation=45)
            st.pyplot(fig)

####################
# Case 2 page layout
####################
if selected == 'Case 2':
    st.title("BMI Prediction")
    st.subheader("Description")

    st.subheader("To predict BMI value, you need to follow the steps below:")
    st.markdown("""
    1. Enter clinical parameters of patient on the left side bar.
    2. Choose the real BMI values you have.
    3. Press the "Predict" button and wait for the result.
    """)
    

    #sidebar layout
    st.sidebar.title("Patiens Info")
    st.sidebar.subheader("Please choose parameters")
    # input features
    sex = st.sidebar.selectbox('Gender', ('Male' , 'Female'))
    age_years = st.sidebar.slider("Age:", min_value = 18, max_value = 100,step = 1)
    asa = st.sidebar.selectbox('ASA Score', ('I', 'II', 'III', 'IV'))
    charleson_comorbidity_index = st.sidebar.selectbox('Charleson Comorbidity Index', (0, 1, 2, 3, 4, 5, 6 , 7))
    depression = st.sidebar.selectbox('Depression', ('Yes', 'No', 'Unknown'))
    prior_abdominal_surgery = st.sidebar.selectbox('Prior Abdominal surgery type', ('Yes', 'No', 'Unknown'))
    diabetes_mellitus_preoperative = st.sidebar.selectbox('Diabetes Mellitus preoperative', ('Yes' , 'No'))
    diabetes_mellitus_after_surgery = st.sidebar.selectbox('Diabetes Mellitus after surgery', ('Yes' , 'No'))
    antidiabetic_drugs_preoperativ___1 = st.sidebar.selectbox('Orale Antidiabetic drugs', ('Yes' , 'No'))
    antidiabetic_drugs_preoperativ___2 = st.sidebar.selectbox('Long-Acting Insulin', ('Yes' , 'No'))
    antidiabetic_drugs_preoperativ___3 = st.sidebar.selectbox('Intermediate-Acting Insulin', ('Yes' , 'No'))
    antidiabetic_drugs_preoperativ___4 = st.sidebar.selectbox('Short-Acting insulin', ('Yes' , 'No'))
    postoperative_hba1c_6_months = st.sidebar.number_input("Postoperative hba1c after 6 month of the surgery:")
    postoperative_hba1c_12_months = st.sidebar.number_input("Postoperative hba1c after 12 month of the surgery:")
    osas_preoperaative = st.sidebar.selectbox('Obstructive Sleep Apnea Syndrome (OSAS) preoperative', ('Yes' , 'No'))
    osas_after_surgery = st.sidebar.selectbox('Obstructive Sleep Apnea Snydrome (OSAS) after surgery', ('Yes' , 'No'))
    surgery = st.sidebar.selectbox('Surgery type', ('Laparoscopy Sleeve Gastrectomy', 'Roux en-Y Gastric Bypass'))
    conversion_f = st.sidebar.selectbox('Conversion from gastric sleeve to gastric bypass', ('Yes', 'No', 'Unknown'))
    surgery_time = st.sidebar.slider("Surgery time (min):", min_value = 25, max_value = 352,step = 1)
    preventive_closure_of_mesenteric_defects = st.sidebar.selectbox('Preventive Closure of Mesenteric Defects type', (1, 2, 3))
    complication___1 = st.sidebar.selectbox('Anastomotic leakage', ('Yes' , 'No'))
    complication___2 = st.sidebar.selectbox('Gastric leakage', ('Yes' , 'No'))
    complication___3 = st.sidebar.selectbox('Intussusception after Roux-en-Y gastric bypass', ('Yes' , 'No'))
    complication___4 = st.sidebar.selectbox('Mesenteric internal hernia after Roux-en-Y gastric bypass', ('Yes' , 'No'))
    complication___5 = st.sidebar.selectbox('Internal hernia through Peterson´s defect after Roux-en-Y gastric bypass', ('Yes' , 'No'))
    complication___6 = st.sidebar.selectbox('Hiatal hernia', ('Yes' , 'No'))
    complication___7 = st.sidebar.selectbox('Gastro Esophageal Refux Disease (GERD)', ('Yes' , 'No'))
    complication___8 = st.sidebar.selectbox('No Complication', ('Yes' , 'No'))
    complication___9 = st.sidebar.selectbox('Anastomotic Ulcera', ('Yes' , 'No'))
    the_clavien_dindo_classification = st.sidebar.selectbox('Clavien-Dindo-Classification', ('I',
                                                                                                               'II',
                                                                                                               'IIIa',
                                                                                                               'IIIb',
                                                                                                               'IVa',
                                                                                                               'IVb',
                                                                                                               'V',
                                                                                                               'No complication'))
    st.subheader("BMI Inputs: ")
    bmi_pre = st.number_input("BMI pre value:")
    # Check box layout to defiine BMI future values
    multi_select_list =  ['6 Months' , '12 Months' , '18 Months',
         '2 Years' , '3 Years' , '4 Years' , '5 Years']
    multi_select_list_2 = ['bmi_pre']
    x = ['bmi_pre']
    multiselect_bmi_future = st.multiselect(
        'Select the real BMI values the patient has',
       multi_select_list)
    # Show the input bmi if they are in the multi select option
    if multi_select_list[0] in multiselect_bmi_future:
        bmi_6_months = st.number_input("Please enter bmi 6 months value:")
        multi_select_list_2.append('bmi_6_months')
    if multi_select_list[0] not in multiselect_bmi_future:
        bmi_6_months = -1
    if multi_select_list[1] in multiselect_bmi_future:
        bmi_12_months = st.number_input("Please enter bmi 12 months value:")
        multi_select_list_2.append('bmi_12_months')
    if multi_select_list[1] not in multiselect_bmi_future:
        bmi_12_months = -1
    if multi_select_list[2] in multiselect_bmi_future:
        bmi_18_months = st.number_input("Please enter bmi 18 months value:")
        multi_select_list_2.append('bmi_12_months')
    if multi_select_list[2] not in multiselect_bmi_future:
        bmi_18_months = -1
    if multi_select_list[3] in multiselect_bmi_future:
        bmi_2_years = st.number_input("Please enter bmi 2 years value:")
        multi_select_list_2.append('bmi_2years')
    if multi_select_list[3] not in multiselect_bmi_future:
        bmi_2_years = -1
    if multi_select_list[4] in multiselect_bmi_future:
        bmi_3_years = st.number_input("Please enter bmi 3 years value:")
        multi_select_list_2.append('bmi_3years')
    if multi_select_list[4] not in multiselect_bmi_future:
        bmi_3_years = -1
    if multi_select_list[5] in multiselect_bmi_future:
        bmi_4_years = st.number_input("Please enter bmi 4 years value:")
        multi_select_list_2.append('bmi_4years')
    if multi_select_list[5] not in multiselect_bmi_future:
        bmi_4_years = -1
    if multi_select_list[6] in multiselect_bmi_future:
        bmi_5_years = st.number_input("Please enter bmi 5 years value:")
        multi_select_list_2.append('bmi_5years')
    if multi_select_list[6] not in multiselect_bmi_future:
        bmi_5_years = -1
    st.subheader("Prediction result: ")
    # Parser user information
    user_input = preprocess_input_case_2(sex, age_years, asa, charleson_comorbidity_index, depression, prior_abdominal_surgery, diabetes_mellitus_preoperative, diabetes_mellitus_after_surgery, antidiabetic_drugs_preoperativ___1, antidiabetic_drugs_preoperativ___2, antidiabetic_drugs_preoperativ___3, antidiabetic_drugs_preoperativ___4, postoperative_hba1c_6_months, postoperative_hba1c_12_months, osas_preoperaative, osas_after_surgery, surgery, conversion_f, surgery_time, preventive_closure_of_mesenteric_defects, complication___1, complication___2, complication___3, complication___4, complication___5, complication___6, complication___7, complication___8, complication___9, the_clavien_dindo_classification, bmi_pre, bmi_6_months, bmi_12_months, bmi_18_months, bmi_2_years, bmi_3_years, bmi_4_years, bmi_5_years)
    
    # Prediction
    predict_button = st.button('Predict')
    if predict_button:
        # Check if all the bmi 6 months, 12 months, etcc are ok
        number_of_warnings = 0
        # Check the len of multiselecth object
        if len(multi_select_list_2) <2:
            st.warning('You are not selecting future BMI values, check the input', icon="⚠️")
            number_of_warnings += 1
        # Check for bmi pre
        if user_input['bmi_pre'].values[0] < 35:
            st.warning('The value of BMI 6 pre is lower than 35, check the input', icon="⚠️")
            number_of_warnings += 1
        # Check for bmi 6 months
        if user_input['bmi_6_months'].values[0] == 0:
            st.warning('The value of BMI 6 months is 0, check the input', icon="⚠️")
            number_of_warnings += 1
        if ((user_input['bmi_12_months'].values[0] != 0) or (user_input['bmi_18_months'].values[0] != 0) or (user_input['bmi_2years'].values[0] != 0) or (user_input['bmi_3years'].values[0] != 0) or (user_input['bmi_4years'].values[0] != 0) or (user_input['bmi_5years'].values[0] != 0)) and user_input['bmi_6_months'].values[0] == 0:
            st.warning('The value of BMI 6 months is 0, check the input', icon="⚠️")
            number_of_warnings += 1
        # Check for bmi 12 months
        if ((user_input['bmi_18_months'].values[0] != 0) or (user_input['bmi_2years'].values[0] != 0) or (user_input['bmi_3years'].values[0] != 0) or (user_input['bmi_4years'].values[0] != 0) or (user_input['bmi_5years'].values[0] != 0)) and user_input['bmi_12_months'].values[0] == 0:
            st.warning('The value of BMI 12 months is 0, check the input', icon="⚠️")
            number_of_warnings += 1
        # Check for bmi 18 months
        if ((user_input['bmi_2years'].values[0] != 0) or (user_input['bmi_3years'].values[0] != 0) or (user_input['bmi_4years'].values[0] != 0) or (user_input['bmi_5years'].values[0] != 0)) and user_input['bmi_18_months'].values[0] == 0:
            st.warning('The value of BMI 18 months is 0, check the input', icon="⚠️")
            number_of_warnings += 1
        # Check for bmi 2 years
        if ((user_input['bmi_3years'].values[0] != 0) or (user_input['bmi_4years'].values[0] != 0) or (user_input['bmi_5years'].values[0] != 0)) and user_input['bmi_2years'].values[0] == 0:
            st.warning('The value of BMI 2 years is 0, check the input', icon="⚠️")
            number_of_warnings += 1
        # Check for bmi 3 years
        if ((user_input['bmi_4years'].values[0] != 0) or (user_input['bmi_5years'].values[0] != 0)) and user_input['bmi_3years'].values[0] == 0:
            st.warning('The value of BMI 3 years is 0, check the input', icon="⚠️")
            number_of_warnings += 1
        # Check for bmi 4 years
        if ((user_input['bmi_5years'].values[0] != 0)) and user_input['bmi_4years'].values[0] == 0:
            st.warning('The value of BMI 4 years is 0, check the input', icon="⚠️")
            number_of_warnings += 1
        if number_of_warnings == 0: # Perform prediction only if there is no warning
            # Predict the single BMI(t+1) value
            columns_to_drop = ['bmi_pre' , 'bmi_6_months','bmi_12_months','bmi_18_months','bmi_2years', 'bmi_3years', 'bmi_4years', 'bmi_5years']
            #pred = model_single.predict(user_input.drop(columns = columns_to_drop))
            #st.header("BMI(t+1):")
            #st.subheader(str(round(pred[0] , 2)) + ' (' + str(round(pred[0] + np.mean(diferences_down) , 2)) + ' ; ' + str(round(pred[0] + np.mean(diferences_up) , 2)) + ')')
            # When the user put some values for bmi 6 months, 12 months, etc...
            # Calculate the predictions using single output model
            # Find the last real value
            if user_input['bmi_6_months'].values[0] == user_input['BMI(t)'].values[0]:
                last_bmi_value = user_input['bmi_6_months']
            elif user_input['bmi_12_months'].values[0] == user_input['BMI(t)'].values[0]:
                last_bmi_value = user_input['bmi_12_months']
            elif user_input['bmi_18_months'].values[0] == user_input['BMI(t)'].values[0]:
                last_bmi_value = user_input['bmi_18_months']
            elif user_input['bmi_2years'].values[0] == user_input['BMI(t)'].values[0]:
                last_bmi_value = user_input['bmi_2years']
            elif user_input['bmi_3years'].values[0] == user_input['BMI(t)'].values[0]:
                last_bmi_value = user_input['bmi_3years']
            elif user_input['bmi_4years'].values[0] == user_input['BMI(t)'].values[0]:
                last_bmi_value = user_input['bmi_4years']
            elif user_input['bmi_5years'].values[0] == user_input['BMI(t)'].values[0]:
                last_bmi_value = user_input['bmi_5years']
            last_bmi_value = pd.DataFrame(last_bmi_value)
            # Calculate the number of predictions we have to do
            iterations = bmi_columns.index(last_bmi_value.columns.tolist()[0])
            pred_single = pd.DataFrame()
            # For loop to predict all future value susing single output model
            for i in range(len(bmi_columns) - iterations - 1):
                if i == 0:
                    # The initial bmi t+1 is based on the last bmi real value
                    aux_prediction = pd.DataFrame(model_single.predict(user_input.drop(columns = columns_to_drop)) , index = [bmi_columns[iterations + i + 1]] , columns = ['BMI Value'])
                    pred_single = pd.concat([pred_single,
                                             aux_prediction] , axis = 0)
                else:
                    # The future bmis will be predicted based on the previous prediction
                    aux_bmi_t = pd.DataFrame({'BMI(t)' : [pred_single.iloc[-1].values[0]]})
                    aux_x = pd.concat([user_input.drop(columns = columns_to_drop + ['BMI(t)']),
                                       aux_bmi_t] , axis = 1)
                    aux_prediction = pd.DataFrame(model_single.predict(aux_x) , index = [bmi_columns[iterations + i + 1]] , columns = ['BMI Value'])
                    pred_single = pd.concat([pred_single,
                                             aux_prediction] , axis = 0)
            # Add the real values to the predictions
            real_bmi_columns = bmi_columns
            real_bmi = user_input[real_bmi_columns].T
            real_bmi.columns = ['BMI Value']
            real_bmi = real_bmi[real_bmi['BMI Value'] != -1]
            real_bmi_columns = real_bmi.index.tolist()
            real_bmi = pd.concat([real_bmi,
                                  pred_single] , axis = 0)
            # Calculate the interval for the predicted values, the rela values don't have
            real_bmi['Lower Difference'] = [0 if i in real_bmi_columns else diferences_down[bmi_columns.index(i)] for i in real_bmi.index.tolist()]
            real_bmi['Upper Difference'] = [0 if i in real_bmi_columns else diferences_up[bmi_columns.index(i)] for i in real_bmi.index.tolist()]
            real_bmi['Lower Bound'] = real_bmi['BMI Value'] + real_bmi['Lower Difference']
            real_bmi['Upper Bound'] = real_bmi['BMI Value'] + real_bmi['Upper Difference']
            real_bmi = real_bmi.drop(columns = ['Lower Difference' , 'Upper Difference'])
            # Create the plot
            st.header("Curve Plot of Evolution of BMI - Single Output Model")
            y = real_bmi['BMI Value'].values.tolist()
            x = real_bmi.index.tolist()
            fig, ax = plt.subplots()
            ax.plot(x[: len(real_bmi_columns)], y[: len(real_bmi_columns)] , color = 'blue' , label = 'Real BMI Values')
            ax.plot(x[len(real_bmi_columns)-1:], y[len(real_bmi_columns)-1:] , color = 'black' , label = 'Predicted BMI Values')
            ax.legend()
            ax.fill_between(
                x, real_bmi['Lower Bound'], real_bmi['Upper Bound'], color='gray', alpha=.15)
            ax.set_title('Evolution of BMI Value')
            ax.set_xlabel('BMI Time Step', 
                       fontweight ='bold')
            ax.set_ylabel('BMI',fontweight ='bold')
            fig.autofmt_xdate(rotation=45)
            st.pyplot(fig)
            # Calculate the predictions using multioutput model
            # Predict all BMI values at once using patients information and bmi pre
            bmi_columns_2 = ['bmi_6_months','bmi_12_months','bmi_18_months','bmi_2years', 'bmi_3years', 'bmi_4years', 'bmi_5years']
            pred_multi = pd.DataFrame(model_multioutput.predict(user_input.drop(columns = bmi_columns_2 + ['BMI(t)'])) , columns = bmi_columns_2).T
            pred_multi.columns = ['BMI Value']
            pred_multi = pd.concat([pd.DataFrame({'BMI Value' : [user_input['bmi_pre'].values[0]]} , index = ['bmi_pre']),
                                    pred_multi] , axis = 0) 
            # Add the real values
            real_bmi_columns = bmi_columns
            real_bmi = user_input[real_bmi_columns].T
            real_bmi.columns = ['BMI Value']
            real_bmi = real_bmi[real_bmi['BMI Value'] != -1]
            real_bmi_columns = real_bmi.index.tolist()
            pred_multi = pd.concat([real_bmi,
                                    pred_multi.iloc[1 + iterations :  , :]] , axis = 0)
            pred_multi['Lower Difference'] = [0 if i in real_bmi_columns else diferences_down[bmi_columns.index(i)] for i in pred_multi.index.tolist()]
            pred_multi['Upper Difference'] = [0 if i in real_bmi_columns else diferences_up[bmi_columns.index(i)] for i in pred_multi.index.tolist()]
            pred_multi['Lower Bound'] = pred_multi['BMI Value'] + pred_multi['Lower Difference']
            pred_multi['Upper Bound'] = pred_multi['BMI Value'] + pred_multi['Upper Difference']
            pred_multi = pred_multi.drop(columns = ['Lower Difference' , 'Upper Difference'])
            # Create the plot
            st.header("Curve Plot of Evolution of BMI - Multi Output Model")
            y = pred_multi['BMI Value'].values.tolist()
            fig, ax = plt.subplots()
            ax.plot(x[: len(real_bmi_columns)], y[: len(real_bmi_columns)] , color = 'blue' , label = 'Real BMI Values')
            ax.plot(x[len(real_bmi_columns)-1:], y[len(real_bmi_columns)-1:] , color = 'black' , label = 'Predicted BMI Values')
            ax.legend()
            ax.fill_between(
                x, pred_multi['Lower Bound'], pred_multi['Upper Bound'], color='gray', alpha=.15)
            ax.set_title('Evolution of BMI Value')
            ax.set_xlabel('BMI Time Step', 
                       fontweight ='bold')
            ax.set_ylabel('BMI',fontweight ='bold')
            fig.autofmt_xdate(rotation=45)
            st.pyplot(fig)

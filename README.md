

#######################################################################################################################################################
# Context:
# This synthetic healthcare dataset has been created to serve as a valuable resource
# for data science, machine learning, and data analysis enthusiasts. 
# It is designed to mimic real-world healthcare data, enabling users to practice, 
# develop, and showcase their data manipulation and analysis skills in the context of the healthcare industry.

# Inspiration:
# The inspiration behind this dataset is rooted in the need for practical and diverse healthcare data for 
# educational and research purposes.
# Healthcare data is often sensitive and subject to privacy regulations, making it challenging to access for 
# learning and experimentation.
# To address this gap, I have leveraged Python's Faker library to generate a dataset that mirrors the structure and attributes 
# commonly found in healthcare records.
# By providing this synthetic data, I hope to foster innovation, learning, and knowledge sharing in the healthcare analytics
# domain.

#Dataset Information:
#Each column provides specific information about the patient, their admission, and the healthcare services provided, making this dataset suitable for various data analysis and modeling tasks in the healthcare domain. Here's a brief explanation of each column in the dataset -

#Name: This column represents the name of the patient associated with the healthcare record.
#Age: The age of the patient at the time of admission, expressed in years.
#Gender: Indicates the gender of the patient, either "Male" or "Female."
#Blood Type: The patient's blood type, which can be one of the common blood types (e.g., "A+", "O-", etc.).
#Medical Condition: This column specifies the primary medical condition or diagnosis associated with the patient, such as "Diabetes," "Hypertension," "Asthma," and more.
#Date of Admission: The date on which the patient was admitted to the healthcare facility.
#Doctor: The name of the doctor responsible for the patient's care during their admission.
#Hospital: Identifies the healthcare facility or hospital where the patient was admitted.
#Insurance Provider: This column indicates the patient's insurance provider, which can be one of several options, including "Aetna," "Blue Cross," "Cigna," "UnitedHealthcare," and "Medicare."
#Billing Amount: The amount of money billed for the patient's healthcare services during their admission. This is expressed as a floating-point number.
#Room Number: The room number where the patient was accommodated during their admission.
#Admission Type: Specifies the type of admission, which can be "Emergency," "Elective," or "Urgent," reflecting the circumstances of the admission.
#Discharge Date: The date on which the patient was discharged from the healthcare facility, based on the admission date and a random number of days within a realistic range.
#Medication: Identifies a medication prescribed or administered to the patient during their admission. Examples include "Aspirin," "Ibuprofen," "Penicillin," "Paracetamol," and "Lipitor."
#Test Results: Describes the results of a medical test conducted during the patient's admission. Possible values include "Normal," "Abnormal," or "Inconclusive," indicating the outcome of the test.
#Usage Scenarios:
#This dataset can be utilized for a wide range of purposes, including:

#Developing and testing healthcare predictive models.
#Practicing data cleaning, transformation, and analysis techniques.
#Creating data visualizations to gain insights into healthcare trends.
#Learning and teaching data science and machine learning concepts in a healthcare context.
#You can treat it as a Multi-Class Classification Problem and solve it for Test Results which contains 3 categories(Normal, Abnormal, and Inconclusive).
#Acknowledgments:
#I acknowledge the importance of healthcare data privacy and security and emphasize that this dataset is entirely synthetic. It does not contain any real patient information or violate any privacy regulations.
#I hope that this dataset contributes to the advancement of data science and healthcare analytics and inspires new ideas. Feel free to explore, analyze, and share your findings with the Kaggle community.

#this data was collected from "https://www.kaggle.com/datasets/prasad22/healthcare-dataset/data"
#although there have been people with similar results there is newer results below. although we are limmited.
#I was able to contribue a bit more to it.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

import warnings
warnings.filterwarnings('ignore')

# Update the file path to reflect the new file name
file_path = r'C:\Users\Michael Lam\Desktop\Hospital fake dataset\healthcaredataset.csv'

# Load the dataset
df = pd.read_csv(file_path)

# Inspect the first few rows
print(df.head())



# Check for missing values
print(df.isnull().sum())

df.tail()

# we can see that the capitlization of the names is problematic and hard to read.
# Convert the 'Name' column to lowercase
df['Name'] = df['Name'].str.lower()

# Display the updated DataFrame
df.head()

#Check the shape of data
print(f'The Training Dataset has {df.shape[0]} rows and {df.shape[1]} columns.')

df.info()

df.isnull().sum()

# we can see that there are no null variables.

# Assuming df is your DataFrame and 'Date of Admission' and 'Discharge Date' are the column names
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])


# Now 'Date of Admission' and 'Discharge Date' columns are converted to datetime format

df.describe()

df.describe(include= "object").T

#Observations:

#Patient Age Range:

#Patients' ages range from 13 to 89 years, with an average age of approximately 52 years.
#Hospital Room Capacity:

#The hospital offers a range of rooms, from 101 to 500, ensuring flexibility in patient accommodation.
#Temporal Coverage:

#Data spans from May 8, 2019, to May 7, 2024, providing a comprehensive five-year view of patient admissions.
#Admission Types:

#Patients enter the hospital through three main admission routes:
#Emergency
#Elective
#Transfer
#Blood Type Distribution:

#Patients exhibit various blood types, with A- being the most prevalent.
#Hospital Distribution:

#The dataset encompasses admissions from 44 hospitals, with LLC Smith being the most frequent.
#Doctor Distribution:

#Among the 27 doctors recorded in the dataset, Michael Smith attends to the highest number of patients.

# Categorical columns
df['Gender'].value_counts()

print(df['Blood Type'].value_counts())

print(df['Admission Type'].value_counts())

print(df['Insurance Provider'].value_counts())

print(df['Doctor'].value_counts().sum())

# Test results analysis
print(df['Test Results'].value_counts())

df.columns

# Plot histogram for the 'Age' column
fig = px.histogram(df, x='Age', title='Age Distribution', nbins=30)
fig.show()

# Define object-type columns
object_columns = ['Gender', 'Blood Type', 'Medical Condition', 'Admission Type', 'Insurance Provider', "Medication", 'Test Results']

# Define pastel color palette
pastel_palette = px.colors.qualitative.Pastel

# Plotly plots for object-type columns
for col in object_columns:
    fig = go.Figure()
    for i, (category, count) in enumerate(df[col].value_counts().items()):
        fig.add_trace(go.Bar(x=[col], y=[count], name=category, marker_color=pastel_palette[i]))
    fig.update_layout(title=f'Distribution of {col}', xaxis_title=col, yaxis_title='Count')
    fig.show()

# Group 'Age' by 'Medical Condition' and calculate the mean age for each condition
age_by_condition = df.groupby('Medical Condition')['Age'].mean().reset_index()

# Plot using Plotly Express with different color palettes
fig = px.bar(age_by_condition, x='Medical Condition', y='Age', color='Medical Condition',
             title='Average Age by Medical Condition',
             labels={'Age': 'Average Age', 'Medical Condition': 'Medical Condition'},
             color_discrete_sequence=px.colors.qualitative.Pastel)
fig.show()

# Group by 'Medical Condition' and 'Medication' and calculate the count for each combination
grouped_df = df.groupby(['Medical Condition', 'Medication']).size().reset_index(name='Count')

# Plot using Plotly Express
fig = px.bar(grouped_df, x='Medical Condition', y='Count', color='Medication', barmode='group',
             title='Medication Distribution by Medical Condition',
             labels={'Count': 'Count', 'Medical Condition': 'Medical Condition', 'Medication': 'Medication'})
fig.show()

# Group 'Sex' by 'Medical Condition' and calculate the count for each combination
sex_by_condition = df.groupby(['Medical Condition', 'Gender']).size().reset_index(name='Count')

# Plot using Plotly Express with different color palettes
fig = px.bar(sex_by_condition, x='Medical Condition', y='Count', color='Gender',
             title='Patient Count by Gender and Medical Condition',
             labels={'Count': 'Patient Count', 'Medical Condition': 'Medical Condition', 'Gender': 'Gender'},
             color_discrete_sequence=px.colors.qualitative.Pastel)
fig.show()

# Group by 'Blood Type' and 'Medical Condition' and calculate the count for each combination
grouped_df = df.groupby(['Blood Type', 'Medical Condition']).size().reset_index(name='Count')

# Plot using Plotly Express
fig = px.bar(grouped_df, x='Blood Type', y='Count', color='Medical Condition', barmode='group',
             title='Patient Count by Blood Type and Medical Condition',
             labels={'Count': 'Patient Count', 'Blood Type': 'Blood Type', 'Medical Condition': 'Medical Condition'})
fig.show()

# Group by 'Blood Type' and 'Gender' and calculate the count for each combination
grouped_df = df.groupby(['Blood Type', 'Gender']).size().reset_index(name='Count')

# Plot using Plotly Express
fig = px.bar(grouped_df, x='Blood Type', y='Count', color='Gender', barmode='group',
             title='Patient Count by Blood Type and Gender',
             labels={'Count': 'Patient Count', 'Blood Type': 'Blood Type', 'Gender': 'Gender'})
fig.show()

# Group by 'Admission Type' and 'Gender' and calculate the count for each combination
grouped_df = df.groupby(['Admission Type', 'Gender']).size().reset_index(name='Count')

# Plot using Plotly Express
fig = px.bar(grouped_df, x='Admission Type', y='Count', color='Gender', barmode='group',
             title='Patient Count by Admission Type and Gender',
             labels={'Count': 'Patient Count', 'Admission Type': 'Admission Type', 'Gender': 'Gender'})
fig.show()

# Group by 'Admission Type' and 'Medical Condition' and calculate the count for each combination
grouped_df = df.groupby(['Admission Type', 'Medical Condition']).size().reset_index(name='Count')

# Plot using Plotly Express
fig = px.bar(grouped_df, x='Admission Type', y='Count', color='Medical Condition', barmode='group',
             title='Patient Count by Admission Type and Medical Condition',
             labels={'Count': 'Patient Count', 'Admission Type': 'Admission Type', 'Medical Condition': 'Medical Condition'})
fig.show()

# Group by 'Test Results' and 'Admission Type' and calculate the count for each combination
grouped_df = df.groupby(['Test Results', 'Admission Type']).size().reset_index(name='Count')

# Plot using Plotly Express
fig = px.bar(grouped_df, x='Test Results', y='Count', color='Admission Type', barmode='group',
             title='Test Results Distribution by Admission Type',
             labels={'Count': 'Count', 'Test Results': 'Test Results', 'Admission Type': 'Admission Type'})
fig.show()

# Group by 'Medication' and 'Gender' and calculate the count for each combination
grouped_df = df.groupby(['Medication', 'Gender']).size().reset_index(name='Count')

# Plot using Plotly Express
fig = px.bar(grouped_df, x='Medication', y='Count', color='Gender', barmode='group',
             title='Medication Distribution by Gender',
             labels={'Count': 'Count', 'Medication': 'Medication', 'Gender': 'Gender'})
fig.show()

#What is the most common blood type among the patients?

most_common_blood_type = df['Blood Type'].value_counts().idxmax()
print(f"The most common blood type among the patients is {most_common_blood_type}.")

How many unique hospitals are included in the dataset?

unique_hospitals = df['Hospital'].nunique()
print(f"There are {unique_hospitals} unique hospitals included in the dataset.")

#Who is the oldest patient in the dataset, and what is their age?

oldest_patient_age = df['Age'].max()
oldest_patient_name = df[df['Age'] == oldest_patient_age]['Name'].iloc[0]
print(f"The oldest patient in the dataset is {oldest_patient_name} with an age of {oldest_patient_age} years.")

#Which doctor has treated the highest number of patients?

doctor_highest_patient_count = df['Doctor'].value_counts().idxmax()
print(f"The doctor who has treated the highest number of patients is {doctor_highest_patient_count}.")

#What is the most frequently prescribed medication?

most_frequent_medication = df['Medication'].value_counts().idxmax()
print(f"The most frequently prescribed medication is {most_frequent_medication}.")

#Are there any seasonal trends in hospital admissions?

# Calculate monthly admissions
monthly_admissions = df['Date of Admission'].dt.month.value_counts().sort_index()

# Create a DataFrame
monthly_admissions_df = pd.DataFrame({'Month': monthly_admissions.index, 'Admissions': monthly_admissions.values})

# Plot the trend using Plotly Express
fig = px.line(monthly_admissions_df, x='Month', y='Admissions', title='Monthly Admissions Trend')
fig.update_xaxes(title='Month')
fig.update_yaxes(title='Number of Admissions')
fig.show()

#What is the average billing amount for patients?

average_billing_amount = df['Billing Amount'].mean()
print(f"The average billing amount for patients is ${average_billing_amount:.2f}.")

#How many male and female patients are there?

male_patients = df[df['Gender'] == 'Male'].shape[0]
female_patients = df[df['Gender'] == 'Female'].shape[0]
print(f"There are {male_patients} Male patients and {female_patients} Female patients.")

#What are the top three most common medical conditions for which patients are admitted?

top_three_medical_conditions = df['Medical Condition'].value_counts().head(3)
print("Top Three Most Common Medical Conditions:")
print("----------------------------------------")
print(top_three_medical_conditions)

###################################################### Let us go futher

# Example Code Enhancements

# Calculate correlation matrix
correlation_matrix = df[['Age', 'Billing Amount', 'Room Number']].corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


#Trend Analysis Example:


# Add a new column for length of stay
df['Length of Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days

# Plot trend of average billing amount over time
df.groupby(df['Date of Admission'].dt.to_period('M')).agg({'Billing Amount':'mean'}).plot()
plt.title('Average Billing Amount Over Time')
plt.xlabel('Date')
plt.ylabel('Average Billing Amount')
plt.show()


##Outlier Detection Example:

# Detect outliers in 'Billing Amount'
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['Billing Amount'])
plt.title('Outliers in Billing Amount')
plt.show()




fig = px.box(df, y='Billing Amount', title='Box Plot of Billing Amount')
fig.show()

#Patient Stay Duration
#Calculate and analyze the length of hospital stay (Discharge Date - Date of Admission) 
#to understand average stay durations and any variations.

df['Length of Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days
fig = px.histogram(df, x='Length of Stay', title='Distribution of Length of Stay')
fig.show()


#Comparative Analysis:
#Compare different hospitals or doctors in terms of patient outcomes, billing amounts, or frequency of conditions.

hospital_comparison = df.groupby('Hospital').agg({'Billing Amount': 'mean', 'Age': 'mean'}).sort_values(by='Billing Amount', ascending=False)
print(hospital_comparison)


#Drawbacks of Using Synthetic Data

# Lack of Real-World Nuances
# Artificial Patterns: Synthetic data might lack the complex patterns and nuances present in real-world data. 
# This could lead to inaccurate models or predictions that do not translate well to real-world scenarios.

# Simplified Distributions: 
# The distributions of synthetic data may be oversimplified compared to actual data,potentially affecting the 
# validity of analysis.

# Potential for Bias:
# Predefined Biases: If the synthetic data generation process includes biases or assumptions, 
#these will be reflected in the data. This could result in models that reinforce existing biases 
# rather than identifying new insights.
# Bias in Data Generation: If the data generation algorithm is biased, it might not accurately 
# reflect the diversity and variability of real-world data.

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv("adult_with_headers.csv")


# # 1. Data Exploration and Preprocessing:

# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe().T


# In[7]:


df.nunique()


# # Since we have only 2 unique values in the income columns that means it is a classification dataset 
# now work according to the classification model 
# there are many classification algorithm
# 1:logistic Regression model
# 2: Decission Treee 
# 3: Random Forest Classsifier
# 4: Support Vector Classifier
#     etc

# In[8]:


min_max=df.copy()


# In[9]:


min_max


# •	Apply scaling techniques to numerical features:
# •	Standard Scaling
# •	Min-Max Scaling
# 

# In[10]:


# first find the numerical columns
numerical=[]
categorical=[]
for col in min_max.columns:
    if min_max[col].dtypes == 'int64':
        numerical.append(col)
    else:
        categorical.append(col)
        


# In[11]:


# This are the numerical columns to apply
numerical


# In[12]:


for i in numerical:
    print(f"unique values of {i} is {min_max[i].unique()}")


# In[13]:


# Required Library
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split


# In[14]:


X=min_max.drop(categorical,axis=1)


# In[15]:


X.nunique()


# In[16]:


Y=min_max['income']


# In[17]:


Y


# In[18]:


X_train,x_test,Y_train,y_test=train_test_split(X,Y,train_size=0.8,random_state=42)


# In[ ]:





# # min _max_scaler
# 
# When to Use:
# Useful for algorithms that rely on distances (e.g., Euclidean) or gradients.
# Works well when data has a uniform distribution and no extreme outliers.
# 
# 
# Suitable for algorithms like:
# K-Nearest Neighbors (KNN)
# Support Vector Machines (SVMs)
# Neural Networks
# 
# 
# Why:
# It ensures all features contribute equally to the model, preventing features with larger scales from dominating.

# In[19]:


#min_max_scaler
min_max_scaler=MinMaxScaler(feature_range=(0,1))
X_train=min_max_scaler.fit_transform(X_train)
print(X_train)


# # Standar Scaler
# Range: Data is scaled to have a mean of 0 and a standard deviation of 1.
# 
# When to Use:
# Useful for algorithms that assume data is Gaussian (normally distributed).
# Works well with algorithms that rely on statistical properties of data.
# 
# Suitable for algorithms like:
# Logistic Regression
# Linear Regression
# Principal Component Analysis (PCA)
# Gradient Descent-based models (e.g., deep learning)
# 
# Why:
# It centers the data and removes the effects of mean and variance, ensuring uniformity across features.

# In[20]:


# Standarization 
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_train


# # 2. Encoding Techniques:
# ### •	Apply One-Hot Encoding to categorical variables with less than 5 categories.
# ### •	Use Label Encoding for categorical variables with more than 5 categories.
# ### •	Discuss the pros and cons of One-Hot Encoding and Label Encoding.
# 

# In[21]:


df.nunique()


# In[22]:


from sklearn.preprocessing import LabelEncoder


# In[23]:


#defining class for the automating data sepration and label encoding 
Encoder=LabelEncoder()
def encoding(df):
    for col in categorical:
        if df[col].nunique()<5 and col!="income":
            df=pd.get_dummies(df,columns=[col])
        else:
            df[col]=Encoder.fit_transform(df[col])
    return df 


# In[24]:


data=encoding(df)


# ## ONE HOT ENCODING
# 
# #### PRONS
# No Ordinality Assumption:
# Does not assume any inherent order in the data, making it ideal for nominal data
# 
# Prevents Misinterpretation:
# Models don’t confuse numerical encoding as ordinal or meaningful relationships between categories.
# 
# Widely Used:
# Works well with most machine learning algorithms, especially those sensitive to numerical magnitude, like linear regression and SVM.
# 
# #### Cons:
# High Dimensionality:
# For a large number of categories, it creates many columns, leading to high memory usage and computational cost (known as the "curse of dimensionality").
# 
# Not Suitable for High Cardinality:
# Inefficient for features with hundreds or thousands of unique categories (e.g., zip codes, user IDs).
# 
# Dummy Variable Trap:
# Correlation between one-hot columns can lead to multicollinearity. This can be avoided by dropping one of the dummy columns.

# ## Label Encoding
# #### Pros:
# 
# Memory Efficient:
# Requires less memory and computation since it replaces categories with integers instead of creating multiple columns.
# 
# Simple and Fast:
# Easy to implement and computationally inexpensive.
# 
# Useful for Ordinal Data:
# Effective for features where categories have a natural order (e.g., "Small," "Medium," "Large").
# 
# #### Cons:
# 
# Assumes Ordinality:
# The model may misinterpret the numerical encoding as ordinal, introducing unintended relationships between categories.
# For example, it may infer that Green (2) is greater than Blue (1), which might not be meaningful for nominal data.
# 
# Poor Performance with Non-Ordinal Data:
# May introduce bias in models sensitive to magnitude (e.g., linear regression, distance-based algorithms like KNN).
# 
# Model Dependency:
# Performs poorly with algorithms that interpret numerical relationships directly, such as tree-based models or neural networks.

# # 3. Feature Engineering:
# ##### •	Create at least 2 new features that could be beneficial for the model. Explain the rationale behind your choices.
# ##### •	Apply a transformation (e.g., log transformation) to at least one skewed numerical feature and justify your choice.
# 

# ### 1. New Capital Gain
# capital_net_gain (Net Capital Gain) (new_feature)
# 
# This feature combines capital_gain and capital_loss into a single meaningful metric.
# 
# A single column capturing net capital gain reduces dimensionality while retaining the information.
# 
# It could highlight individuals with high net financial benefits, which may correlate with their income level.
# 
# ###  2. work_hours_category (Work Hours Category)
# 
# Categorize hours_per_week into meaningful bins:
# 
# <20: "Part-time"
# 
# 20–40: "Full-time"
# 
# >40: "Overtime"
# 
# Income is often related to the number of hours worked.
# 
# Categorizing work hours can help the model identify patterns in work behavior (e.g., people working overtime might have higher incomes).
# 
# It can be especially helpful if hours_per_week exhibits non-linear relationships with income

# In[25]:


data['Net_Capital_Gain']=df['capital_gain']-df["capital_loss"]


# In[26]:


data


# In[27]:


def time_category(hours):
    if hours <=20:
        return "Part Time"
    elif 20<=hours<=40:
        return "Full Time"
    else:
        return "Overtime"
    
    
data['Work_hour_category']=df["hours_per_week"].apply(time_category)


# In[28]:


data


# In[29]:


data=pd.get_dummies(data,columns=["Work_hour_category"])


# In[30]:


data


# # 4. Feature Selection:
# •	Use the Isolation Forest algorithm to identify and remove outliers. Discuss how outliers can affect model performance.
# 
# •	Apply the PPS (Predictive Power Score) to find and discuss the relationships between features. Compare its findings with the correlation matrix.
# 

# In[31]:


target=data.drop('income',axis=1)


# In[33]:


target


# In[34]:


data


# In[35]:


from sklearn.ensemble import IsolationForest
iso_forest=IsolationForest(contamination=0.05, random_state=42)
outliers = iso_forest.fit_predict(data)


# In[36]:


data['outlier']=outliers


# In[37]:


data


# In[38]:


# Remove outliear 
data_cleaned = data[data['outlier'] == 1].drop(['outlier'], axis=1)


# In[39]:


data_cleaned


# Outliers can distort the mean, variance, and other summary statistics, leading to biased parameter estimates in models like linear regression

# In[40]:


import ppscore as pps
import seaborn as sns
import matplotlib.pyplot as plt
# Calculate the PPS matrix
pps_matrix = pps.matrix(data_cleaned)

# Visualize the PPS 
#matrix
plt.figure(figsize=(15,8))
sns.heatmap(pps_matrix.pivot(index='x', columns='y',values='ppscore'), annot=True, cmap='coolwarm')
plt.title('Predictive Power Score Matrix')
plt.show()


# In[41]:


z=data_cleaned.corr()

plt.figure(figsize=(15,8))
sns.heatmap(z,annot=True)
plt.show()


# # PPS vs. Correlation:
# PPS provides a more comprehensive measure of feature relationships, capturing non-linear dependencies and handling mixed data types.
# Correlation is limited to linear relationships and numerical data but is computationally simpler.

# In[ ]:





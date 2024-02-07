#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR


# In[2]:


data=pd.read_csv("D:\Downloads\engineering colleges in India (1).csv")


# In[3]:


data = pd.read_csv("D:\Downloads\engineering colleges in India (1).csv", low_memory=False)


# In[4]:


data.head()


# In[5]:


data.info()


# In[6]:


students="Total Student Enrollments"
teachers="Total Faculty"


# In[7]:


data[students] = pd.to_numeric(data[students], errors='coerce')
data[teachers] = pd.to_numeric(data[teachers], errors='coerce')


# In[8]:


data["student_to_faculty_ratio"]=data[students]/data[teachers]


# In[9]:


data.head()


# In[10]:


data["student_to_faculty_ratio"]


# In[11]:


data.info()


# In[12]:


data.describe()


# In[13]:


data=data.loc[:, ~data.columns.str.contains('^Unnamed')]


# In[14]:


data.head()


# In[15]:


data.describe()


# In[16]:


data.info()


# In[17]:


features=["College Name","student_to_faculty_ratio","University","Facilities","City","College Type","Rating"]
df_cp=data[features].copy()


# In[18]:


df_cp.head()


# In[19]:


df_cp.info()


# In[45]:


#feature_drop=["University","College Name"]
df_cp.drop(columns=["University"], inplace=True)


# In[46]:


df_cp.drop(columns=["College Name"], inplace=True)
df_cp.info()


# In[47]:


df_clean = df_cp.dropna(subset=['Rating'])


# In[48]:


df_clean.info()


# In[51]:


X = df_clean.drop(columns=['Rating'])
y = df_clean['Rating']
X


# In[52]:


#X = pd.get_dummies(X)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Drop samples with missing values
X_train_cleaned = X_train.dropna()

# You may need to drop corresponding labels in y_train as well if there are missing values
y_train_cleaned = y_train.drop(y_train.index.difference(X_train_cleaned.index))

X_valid_cleaned = X_valid.dropna()

# You may need to drop corresponding labels in y_train as well if there are missing values
y_valid_cleaned = y_valid.drop(y_valid.index.difference(X_valid_cleaned.index))



# In[53]:


print("Training dataset feature names:", X_train_cleaned.columns)
X_train_cleaned


# In[54]:


#svr_model = SVR(kernel='rbf', C=100, epsilon=0.1)

# Train the SVR model
#svr_model.fit(X_train_cleaned, y_train_cleaned)

# Make predictions on the validation set
#y_pred_svr = svr_model.predict(X_valid_cleaned)

# Calculate RMSE (Root Mean Squared Error) to evaluate the model performance
#rmse_svr = sqrt(mean_squared_error(y_valid_cleaned, y_pred_svr))
#print("RMSE for SVR:", rmse_svr)

model = RandomForestRegressor(random_state=42)
model.fit(X_train_cleaned, y_train_cleaned)
X_valid_cleaned = X_valid.dropna()
y_valid_cleaned = y_valid.drop(y_valid.index.difference(X_valid_cleaned.index))

# Now, make predictions on the cleaned validation set
y_pred = model.predict(X_valid_cleaned)

# Calculate RMSE (Root Mean Squared Error) to evaluate the model performance
rmse = sqrt(mean_squared_error(y_valid_cleaned, y_pred))
print(f"RMSE: {rmse}")


# In[40]:


df_main_subtracted = df_cp.drop(df_clean.index, errors='ignore')


# In[41]:


df_main_subtracted


# In[30]:


X_test=df_main_subtracted.drop(columns='Rating')


# In[31]:


X_test = pd.get_dummies(X_test)
X_test_aligned, _ = X_test.align(X_train_cleaned, axis=1, fill_value=0)
#print(X_test_aligned.columns)
#X_test_aligned


# In[32]:


predictions=model.predict(X_test_aligned)


# In[34]:


X_test=X_test[X_train_cleaned.columns]
missing_columns = set(X_train_cleaned.columns) - set(X_test.columns)
for column in missing_columns:
    X_test[column] = 0


# In[ ]:


# Get the list of column names
columns = X_test_aligned.columns.tolist()

# Remove the last_column_name from the list of columns
columns.remove("student_to_faculty_ratio")

# Insert the last_column_name at the second position
columns.insert(1, "student_to_faculty_ratio")

# Reorder columns
X_test_aligned = X_test_aligned[columns]


# In[ ]:


predictions=svr_model.predict(X_test_aligned)


# In[ ]:





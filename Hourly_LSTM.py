#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing necessary python libraries
import os
import pandas as pd
import tqdm
from tqdm import tqdm_notebook #This just visualizes the green progress bar that you see during training
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential,load_model #Building a sequential Keras model that is a linear stack of layers 
from keras.layers import LSTM, Dense
from keras.layers import Dropout, Activation, Flatten
from keras.optimizers import SGD


# In[3]:


os.getcwd()
os.chdir("./Aggregated_Data/Hourly")
df_ge = pd.read_csv("Sub_20.csv", index_col=0) 
print(df_ge.head())
os.chdir("../../")
print(os.getcwd())


# In[68]:


print("checking if any null values are present\n", df_ge.isna().sum())


# In[69]:


#Specify the training columns by their names
train_cols = ["Precipitation","Temperature","sin_day","cos_day","Month","Three_Day_Flow","Two_Week_Flow", "Month_Flow", "Season_Flow","Precip_Times_Temperature","Temperature_Times_Day","Precip_Times_Temperature_Times_Day"]
label_cols = ["Outflow"]

#Specify the number of lagged hours that you are interested in 
TIME_STEPS = 18

df_train, df_test = train_test_split(df_ge, train_size=0.6, test_size=0.4, shuffle=False)
print("Train and Test size", len(df_train), len(df_test))


# In[70]:


# Loading training data and labels into x_train and y_train respectively
x = df_train.loc[:,train_cols].values
y = df_train.loc[:,label_cols].values
x_train=x
y_train=y
#Load testing data and labels into x_test and y_test
x_test = df_test.loc[:,train_cols].values
y_test = df_test.loc[:,label_cols].values
print(x_test)


# In[71]:


# This function normalizes the input data
def Normalization_Transform(x):
    x_mean=np.mean(x, axis=0)
    x_std= np.std(x, axis=0)
    xn = (x-x_mean)/x_std
    return xn, x_mean,x_std


# In[ ]:


# This function reverses the normalization 
def inverse_Normalization_Transform(xn, x_mean,x_std):
    xd = (xn*x_std)+x_mean
    return xd


# In[ ]:


#This function creates the time series of input data that will feed into the LSTM. Every time step you will feed in a matrix
#of inputs that is (length of time step)x(number of input variables). For example, if you're using a lag of 18 hours for only precip
#and temperature, but you just want the present values for the other variables, you will have a matrix that is 18x12, where 12 is the 
#number of input variables. The first two columns will be completely populated with the previous 18 hours of values, but the rest of the columns will only have the
#first row populated and the rest of the rows are zero. 

# If you open this jupyter notebook in Spyder, it's much easier to visualize what's going on here. 
def build_timeseries(mat, y_label,y_label_Actual, TIME_STEPS):
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,))
    y_Actual = np.zeros((dim_0,))
    
    for i in tqdm_notebook(range(dim_0)):
        x[i] = mat[i:TIME_STEPS+i]
        y[i] =y_label[TIME_STEPS+i, 0]
        y_Actual[i] =y_label_Actual[TIME_STEPS+i, 0]
        x[i,1:,2:]=0
    print("length of time-series i/o",x.shape,y.shape)
    return x, y, y_Actual


# In[ ]:


#This block of code just uses the above-defined functions to normalize and build the time series of data

# Normalizing Data Training Data
x_train_Nor , x_mean_train, x_std_train = Normalization_Transform(x_train)
y_train_Nor , y_mean_train, y_std_train = Normalization_Transform(y_train)

# Converting the data into timeseries 
X_Train, Y_Train, Y_Train_Actual = build_timeseries(x_train_Nor, y_train_Nor,y_train,TIME_STEPS)

# Normalizing Validation  Data
x_test_Nor , x_mean_test, x_std_test = Normalization_Transform(x_test)
y_test_Nor , y_mean_test, y_std_test = Normalization_Transform(y_test)

# Converting the data into timeseries 
X_Val, Y_Val,Y_Val_Actual = build_timeseries(x_test_Nor,y_test_Nor, y_test, TIME_STEPS)

print(X_Train)


# In[ ]:


print(X_Val)
print(Y_Val)


# In[ ]:


# Initialize the LSTM as a sequential model, so we're just going to keep tacking on layers 
model = Sequential() 
# Adding the first LSTM layer and some dropout for regularization (to minimize overfitting)
#The first layer always needs to receive an input shape which is defined by the training data
#Adding return_sequences=True is necessary for stacking LSTM layers (if you have an LSTM layer after another LSTM layer)
model.add(LSTM(24, return_sequences=True,
               input_shape=(X_Train.shape[1], X_Train.shape[2])))  
model.add(Dropout(0.2))

# Adding a couple more LSTM layers and dropout regularization. The units (hidden states), the number of layers, and the dropout regularization
#term are all hyperparameters that you will want to optimize. The values here are just typical default values to start off with. 

model.add(LSTM(units = 50,return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50))
model.add(Dropout(0.2))

#Usually you need to just implement a dense layer to serve as the output layer, but I added an additional layer here 
#because I got better results

model.add(Dense(20,activation='relu')) #RELU is a more common activation function now rather than sigmoidal. 


#This last dense layer is just the output layer, which has a linear activation function because you're essentially
#just returning the results from the last layer. 

model.add(Dense(1, activation="linear"))



#This line is where you compile the final model. We are using the adam (default) gradient descent based optimizer
#The choice of loss function is based on your understanding of the inherent noise in the data. 
#You can also implement a custom loss function. 
#I also implemented an accuracy metrics but in the context of regression it doesn't mean anything (only in classification). 

model.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics=['accuracy']) 

# The batch size and number of epochs are also hyperparameters that can be tuned. 
#Generally, you will want to have a large number of epochs, but over you'll probaby see only marginal benefits
#The batch size can kind've have a large implication on how the LSTM trains. Batches can maybe act as a 
#proxy to train for seasonality. Typical batch sizes are closer to 24, 48, 64, but I found that larger batch sizes 
#might find some seasonality
history=model.fit(X_Train, Y_Train,
          batch_size=1000, epochs=1,
          validation_data=(X_Val, Y_Val))
# Model summary for number of parameters use in the algorithm 
model.summary()


# In[ ]:



# In[ ]:


#Check if inversion is working correctly



# In[ ]:


#Prediction of validation data labels. You predict the normalize value and then invert it 
predict =inverse_Normalization_Transform( model.predict(X_Val),y_mean_train, y_std_train)

#predict


# In[ ]:


#Plot the prediction results and the validation results



# In[ ]:


#mean_squared_error(predict, Y_Val_Actual)
print(predict)


# In[ ]:


import numpy
#os.chdir("./Hourly_Predictions")
print(os.getcwd())
numpy.savetxt("Sub20_60_40.csv", np.c_[predict,Y_Val_Actual], delimiter=",") #Currently have to do this manually



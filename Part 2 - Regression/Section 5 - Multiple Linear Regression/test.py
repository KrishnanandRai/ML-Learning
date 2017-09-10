class linear_model:
   
    def __init__(self):
         pass

    def fit(self, X, y, X_test):
        X = np.append(arr = np.ones((X.shape[0],1)).astype(int), values = X, axis = 1)
        X_test = np.append(arr = np.ones((X_test.shape[0],1)).astype(int), values = X_test, axis = 1)
        print(X.shape[1])
        theta = np.zeros(X.shape[1])
        
        self.grad_desc(X, y, theta)
        z = self.predict(X_test,theta)
        print(z)
    
    
    def grad_desc(self, X, y, theta):
        beta = np.zeros(X.shape[1])
        m = X.shape[1]
        while (1) :
        
            for i in range(m):
                
                beta[i] = theta[i] - (0.01/50)*(((X.dot(np.matrix(theta).T) - y )*(X[:,i:i+1])).sum())
               
            if (theta - beta).all() < 10**-3:
                break
            theta[:] = beta[:]
            #print(theta)
        
        
    def predict(self, X, theta):
            return (X.dot(np.matrix(theta).T))
        
        
        
# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#encoding catagorical data state 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, -1] = labelencoder_X.fit_transform(X[:, -1])

onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoind the dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
 
    
lin_mod= linear_model()
lin_mod.fit(X_train, y_train, X_test)
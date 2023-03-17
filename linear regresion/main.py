# *Importing the libraries
import numpy as np

# the pyplot module of the matplotlib will be used for vidualisation alternatively pyplot can be used 
import matplotlib.pyplot as plt

# pandas for reading the data and performing pre analysis
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')

# extracting all the independent variables which in this case is just one 
X = dataset.iloc[:, :-1].values

# geting hold of the independent variables 
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#training data 
print(X_train)
print(y_train)

# test data 
print(X_test)
pprint(y_test)

from sklearn.linear_model import LinearRegression

# creating an object of the LinearRegression class
regressor = LinearRegression()

#training the data with the fit method from the LinearRegression class
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
#ploting the actuall traing data 
plt.scatter(X_train, y_train, color = 'red')

#ploting a straight line of the trained data 
plt.plot(X_train, regressor.predict(X_train), color = 'blue')

#plot configurations
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

#displaying the plot on screen
plt.show()

# Visualising the Test set results
#plot the test set 
plt.scatter(X_test, y_test, color = 'red')

#ploting a straight line of the trained data
plt.plot(X_train, regressor.predict(X_train), color = 'blue')

#plot configurations
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

#displaying the plot
plt.show()

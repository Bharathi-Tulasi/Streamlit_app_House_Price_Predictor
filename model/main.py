import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pickle

def get_cleaned_data():

  #read the data file
  data = pd.read_csv("data/Housing.csv")
  
  # Get dummy values for categorical features
  data = pd.get_dummies(data = data)

  

  return data

def create_model(data):
  # input and output
  X = data.drop(["price", "mainroad_no", "guestroom_no", "basement_no", "airconditioning_no", "prefarea_no", "hotwaterheating_no", "furnishingstatus_semi-furnished", "furnishingstatus_unfurnished"], axis = 1)
  y = data["price"]
  

  # Variable Inflation Factor (VIF) to check multicollinearity and to drop unnecessary features
  # 'X' is the DataFrame of independent variables
  vif = pd.DataFrame()
  vif["Variable"] = X.columns
  vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

  print("VIF", vif)
  print(X.head)

  #scaling
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X.values)

  # split the data
  x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 42)

  # train the model
  model = LinearRegression()
  model.fit(x_train, y_train)
  r2 = model.score(x_train, y_train)
  n = x_train.shape[0]
  p = x_train.shape[1]
  adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)
  print("r-score: ", r2)
  print("adjusted_r2: ", adj_r2)

  
  # test the model
  y_pred = model.predict(x_test)
  
  
  return model, scaler





def main():
  data = get_cleaned_data()
  model, scaler = create_model(data)
  
  with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)
  with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)


if __name__ == '__main__':
  main()
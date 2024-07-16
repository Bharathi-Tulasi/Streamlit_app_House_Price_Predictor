
import pickle
import pandas as pd
import numpy as np
import streamlit as st

def get_cleaned_data():
  data = pd.read_csv("data/Housing.csv")
  
  data = pd.get_dummies(data = data)
  return data


def col1_input(input_dict1):
  
  st.subheader("Select your options")
  data = get_cleaned_data()

  num_labels = [("Area", "area"),
                   ("Number of Bedrooms", "bedrooms"),
                   ("Number of Bathrooms", "bathrooms"),
                   ("Number of stories", "stories"), 
                   ("Parking slots", "parking")
               ]
    
    

  for key, value in num_labels:
    input_dict1[key] = st.number_input(
        key,
        min_value = data[value].min(),
        max_value = data[value].max()
      )
  
  return input_dict1


def col2_input(input_dict2):
  st.subheader("Select your options")
  

  checkbox_labels = [("Mainroad", "mainroad_yes"), 
                    ("Guestroom", "guestroom_yes"), 
                    ("Basement", "basement_yes"), 
                    ("Hot water Heating", "hotwaterheating_yes"), 
                    ("Airconditiong", "airconditioning_yes"),
                    ("Preferred Area", "prefarea_yes"),
                    ("Furnished", "furnishingstatus_furnished")
                  ]

    

  for key, value in checkbox_labels:
    input_dict2[value] = st.checkbox(key)
    input_dict2 = {key: int(values) for key, values in input_dict2.items()}
    
  return input_dict2  


def do_predictions(input_data):
  model = pickle.load(open("model/model.pkl", "rb"))
  scaler = pickle.load(open("model/scaler.pkl", "rb"))

  input_arr = np.array(list(input_data.values())).reshape(1,-1)
  input_arr_scaled = scaler.transform(input_arr)

  prediction = model.predict(input_arr_scaled)
  
  st.header(int(prediction[0]))



def main():
  st.set_page_config(
  page_title = "House Price Predictor",
  page_icon = ":house_with_garden:",
  layout = "wide",
  initial_sidebar_state = "expanded"

  ) 

  with st.container():
    st.title("House Price Predictor")
    st.header("Know the approximate price of your dream house!")
   
   
  col1, col2, col3  = st.columns([1, 1, 1])
   
  
  
  input_dict1 = {}
  input_dict2 = {}

  with col1:
    input1 = col1_input(input_dict1)
      
  

  with col2:
    input2 = col2_input(input_dict2)
    
    


  input_data = {**input1, **input2}
  
  

  with col3:
    if st.button("**Predict My Dream House Price**", type = "primary", use_container_width = True):
      do_predictions(input_data) 
    

    
if __name__ == "__main__":
    main()
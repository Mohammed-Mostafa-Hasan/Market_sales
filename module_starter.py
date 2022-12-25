# ------- BEFORE STARTING - SOME BASIC TIPS
# You can add a comment within a Python file by using a hashtag '#'
# Anything that comes after the hashtag on the same line, will be considered
# a comment and won't be executed as code by the Python interpreter.

# --- 1) IMPORTING PACKAGES
# The first thing you should always do in a Python file is to import any
# packages that you will need within the file. This should always go at the top
# of the file
import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import data_extraction


# --- 2) DEFINE GLOBAL CONSTANTS
# Constants are variables that should remain the same througout the entire running
# of the module. You should define these after the imports at the top of the file.
# You should give global constants a name and ensure that they are in all upper
# case, such as: UPPER_CASE
K = 10 
"""
URL_ROOT = "https://github.com/Mohammed-Mostafa-Hasan/Market_sales/raw/main/"
PATH = os.path.join("Resources","MarketData")
URL_SALES = URL_ROOT + "Resources/sales.rar"
URL_STOCK = URL_ROOT + "Resources/sensor_stock_levels.rar"
URL_TEMP  = RUL_ROOT + "Resources/sensor_storage_temperature.rar"
"""
# --- 3) ALGORITHM CODE
# Next, we should write our code that will be executed when a model needs to be 
# trained. There are many ways to structure this code and it is your choice 
# how you wish to do this. The code in the 'module_helper.py' file will break
# the code down into independent functions, which is 1 option. 
# Include your algorithm code in this section below:
#all steps before model traning should be included
"""
-for data cleaning and data collection 
-we can use these steps
-but we prepare all these steps in seperated file called practice
-within the same folder

    

data_extraction.fetch_data(URL_SALES,PATH,"sales.rar")
data_extraction.fetch_data(URL_STOCK,PATH,"sensor_stock_levels.csv")
data_extraction.fetch_data(URL_TEMP,PATH,"sensor_storage_temperature.csv")


sales_df = get_data("sales.csv")
stock_df = get_data("sensor_stock_levels.csv")
temp_df  = get_data("sensor_storage_temperature.csv") 

sales_df.info()
stock_df.info()
temp_df.info()

drop unwanted columns
sales_df.drop(columns = ["Unnamed: 0"], inplace=True, error='ignore')
stock_df.drop(columns = ["Unnamed: 0"], inplace=True, error='ignore')
temp_df.drop(columns = ["Unnamed: 0"], inplace=True, error='ignore')
"""
#obtain prepared data for ML model
market_df = pd.read_csv("Data/cleaned_data.csv")
#create target variabel and predicted variable
def  create_target_independed_vars(
     data:pd.DataFrame=None,
     target:str = 'estimated_stock_pct' 
):  
    #check if target variable within the data
    if target not in data.columns:
        raise Exception(f"target variable: {target} not in data")
        
    X = data.drop(columns = data[target])
    y = data[target]
    return X,y
    
# --- 4) MAIN FUNCTION
# Your algorithm code should contain modular code that can be run independently.
# You may want to include a final function that ties everything together, to allow
# the entire pipeline of loading the data and training the algorithm to be run all
# at once

#train the algorithm using cross validation
X, y = create_target_independed_vars(market_df,'estimated_stock_pct')
def train_model_with_cv(
    X:pd.DataFrame=None,
    y:pd.Series=None
):
    for fold in range(0,K):
             
        accuracy = []
        X_train, X_test, y_train, y_test  = train_test_split(X, y, train_size = 0.75 , random_state = 42 )
        scaller = StandardScaler()
        rf_model = RandomForestRegressor()
        scaller.fit(X_train)
        X_train = scaller.transform(X_train)
        y_train = scaller.transform(y_trian)
        model = rf_model.fit(X_train, y_train)
        pred = model.predict(X_test)            
        mae = mean_absolute_error(y_test, pred)  
        accuracy.append(mae)
        print(f"cross validation #{fold+1} and MSE is: {mae:.3f} ")
    print(f"Average MAE: {(sum(accuracy) / len(accuracy)):.2f}")









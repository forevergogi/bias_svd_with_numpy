import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def csv2arr(file_path):
    df = pd.read_csv(file_path)
    df_dropped = df.drop(columns=['ctime'])
    arrs = df_dropped.values
    for i in range(arrs.shape[0]):
        status = arrs[i][2]
        arrs[i][2] = 1 if status == 0 else 0
    return arrs

def train_test_val_split(mat):
    #Approximate ratio is 0.6:0.2:0.2
    train_set,test_set = train_test_split(mat,test_size=0.2,random_state=42)
    train_set,val_set = train_test_split(train_set,test_size=0.33,random_state=41)
    return train_set,test_set,val_set

def get_data(file_path='data/new_all_order.csv'):
    arrs = csv2arr(file_path)
    return train_test_val_split(arrs)










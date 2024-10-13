#!/usr/bin/env python
import pandas as pd
import numpy as np
import dataframe_image as dfi
import os
from pathlib import Path

# create a custom function
def calculate_sensitivity(y, y_pred):
    
    tp, tn, fn, fp = 0.0,0.0,0.0,0.0
    for l,m in enumerate(y):        
        if m==y_pred[l] and m==1:
            tp+=1
        if m==y_pred[l] and m==0:
            tn+=1
        if m!=y_pred[l] and m==1:
            fn+=1
        if m!=y_pred[l] and m==0:
            fp+=1
            
    return tp/(tp+fn)

# create a custom function
def calculate_specificity(y, y_pred):
    
    tp, tn, fn, fp = 0.0,0.0,0.0,0.0
    for l,m in enumerate(y):        
        if m==y_pred[l] and m==1:
            tp+=1
        if m==y_pred[l] and m==0:
            tn+=1
        if m!=y_pred[l] and m==1:
            fn+=1
        if m!=y_pred[l] and m==0:
            fp+=1
            
    return tn/(tn+fp)

def save_df_to_png(df, path):
    df_styled = df.style.background_gradient()
    dfi.export(df_styled,path)
    print(f"Stored df in {path}")



def load_all_data(params_file):

    # Location of feature dataset
    dataset_folder = Path(os.getcwd()+'/dataset')
    feature_dataset_filename = dataset_folder/Path('features')/Path(f"{params_file}.pkl")
    df_radiomics = pd.read_pickle(feature_dataset_filename)
    print("Radiomics Dataset shape", df_radiomics.shape)


    # Load the clinical_data
    dataset_folder = Path(os.getcwd()+'/dataset')
    clinical_data = pd.read_excel(dataset_folder/Path('Pretreat-MetsToBrain-Masks_clin_20230918.xlsx'),sheet_name='Data')
    clinical_data = clinical_data.dropna(axis=0, subset=['Death'])
    clinical_data.Death = clinical_data.Death.astype(int)
    print("Clinical Data Dataset shape", clinical_data.shape)

    # Select and process the clinical_data
    target_labels = clinical_data[['BraTS_MET_ID', 'Death']]

    ### Merge features with clinical data
    #### Create 2 new datasets, 
    # - 1. one with only the features + target label
    # - 2. one with the features + all clinical data + target label

    df1 = pd.merge(df_radiomics, target_labels, how='left', on='BraTS_MET_ID')
    df1 = df1.drop('BraTS_MET_ID', axis=1)
    df1 = df1.dropna(axis=0, subset=['Death'])
    df1["Death"] = df1["Death"].astype(int)
    print("Dataset with Radiomics features and target label shape", df1.shape)

    df2 = pd.merge(df_radiomics, clinical_data, how='left', on='BraTS_MET_ID')
    df2 = df2.drop('BraTS_MET_ID', axis=1)
    df2 = df2.dropna(axis=0, subset=['Death'])
    df2["Death"] = df2["Death"].astype(int)
    print("Dataset with Radiomics features and clinical data shape", df2.shape)

    return df1, df2



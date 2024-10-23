#!/usr/bin/env python
import pandas as pd
import numpy as np
import dataframe_image as dfi
import os
from pathlib import Path
import matplotlib.pyplot as plt

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


def plot_top_models(metrics, models, scores, experiment_name):
    # Custom colors for each model

    colors = [
        '#003f5c',  # Darkest blue
        '#2f4b7c',  # Dark blue
        '#665191',  # Purple-blue
        '#a05195',  # Violet-blue
        '#d45087',  # Light violet-blue
        '#f95d6a',  # Pinkish-blue
    ]
    # colors = [
    #     'midnightblue',  # Darkest blue
    #     'mediumblue',  # Dark blue
    #     'dodgerblue',  # Light violet-blue
    #     'mediumpurple',  # Pinkish-blue
    #     'blueviolet',  # Purple-blue
    #     'cyan',  # Violet-blue
    # ]
    cmap={}
    for c,m in zip(colors,models):
        cmap[m]=c

    # Set up bar width and positions
    bar_height = 0.12
    index = np.arange(len(metrics))

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))

    models_revered = models[::-1]
    # Plot each model's performance across the metrics using horizontal bars
    for i, model in enumerate(models_revered):
        ax.barh(index + i * bar_height, scores[model], bar_height, label=model, color=cmap[model])

    # Add labels, title, and legend
    ax.set_ylabel('Metric')
    ax.set_xlabel('Score')
    ax.set_title(f'Comparison of Models Performance - Experiment: {experiment_name}')
    ax.set_yticks(index + bar_height * 2.5)
    ax.set_yticklabels(metrics)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    ax.set_xlim(0, 0.8)
    # Add values on top of each bar
    for i, model in enumerate(models_revered):
        for j, score in enumerate(scores[model]):
            ax.text(score + 0.01, index[j] + i * bar_height, f'{score:.2f}', va='center', ha='left', fontsize='x-small')

    # Display the plot
    plt.savefig(f'./results/{experiment_name}/model_performance.png', bbox_inches="tight") 
    plt.tight_layout()
    plt.show()


MODEL_PLOTS = {
    'pipeline' : 'Schematic drawing of the preprocessing pipeline',
    'auc' : 'Area Under the Curve',
    'threshold' : 'Discrimination Threshold',
    'pr' : 'Precision Recall Curve',
    'confusion_matrix' : 'Confusion Matrix',
    'error' : 'Class Prediction Error',
    'class_report' : 'Classification Report',
    'boundary' : 'Decision Boundary',
    'rfe' : 'Recursive Feature Selection',
    'learning' : 'Learning Curve',
    'manifold' : 'Manifold Learning',
    'calibration' : 'Calibration Curve',
    'vc' : 'Validation Curve',
    'dimension' : 'Dimension Learning',
    'feature' : 'Feature Importance',
    'feature_all' : 'Feature Importance (All)',
    'parameter' : 'Model Hyperparameter',
    'lift' : 'Lift Curve',
    'gain' : 'Gain Chart',
    'tree' : 'Decision Tree',
    'ks' : 'KS Statistic Plot',
}

#(Transformed train set shape, Transformed test set shape, Preprocess, Imputation type, Numeric imputation, Categorical imputation, CPU Jobs, Log Experiment, USI)
rows_to_drop = [5, 6, 8, 9, 10, 11, 16, 18, 20] # basic
rows_to_drop = [5, 6, 8, 9, 10, 11, 18, 20, 22] # remove_outliers
rows_to_drop = [5, 6, 8, 9, 10, 11, 20, 22, 24] # remove_multicollinearity
rows_to_drop = [5, 6, 8, 9, 10, 11, 22, 24, 26] # fix_imbalance_synthetic_data
rows_to_drop = [5, 6, 8, 9, 10, 11, 25, 27, 29] # pca
rows_to_drop = [5, 6, 8, 9, 10, 11, 29, 31, 33] # feature_selection
rows_to_drop = [5, 6, 8, 9, 10, 11, 18, 20] # ensemble_models
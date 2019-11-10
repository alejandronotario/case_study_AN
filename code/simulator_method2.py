#!/usr/bin/env python

#title               :predictions simulator
#description         :KNN model building to predict the frequency of claims a customer’s going to make with the company
#author              :Alejandro Notario
#date                :2019-11-07
#version             :
#usage               :simulator.py
#notes               :
#requirements        :Libraries in this script
#python version      :3.6
#==================================================================================

#Libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')



neighbors=input("Please enter the neighbors to train the model: ")
neighbors=int(neighbors)

def data_cleaning():
    """This function returns the data cleaned and
    ready to run the model
    """
    df=pd.read_csv('casestudy_data.csv', sep =',', low_memory=False)
    def recode(row):
        if row['claim_count'] >= 2: 
            val = 2
        else:
            val=row['claim_count']

        return val
    df['target'] = df.apply (recode, axis=1)
    df['num_exposure_num'] = pd.to_numeric(df['num_exposure'], errors='coerce')
    df = df[~df['num_exposure_num'].isnull()]
    df['num_driverAge_num'] = pd.to_numeric(df['num_driverAge'], errors='coerce')
    df.num_driverAge_num=df.num_driverAge_num.fillna(df.num_driverAge_num.median())
    df.cat_fuelType=df.cat_fuelType.fillna("Regular")
    def recode2(row):
        if row['ord_vehicleHP'] == 1111:
            val = 16
        elif row['ord_vehicleHP'] == 9999:
            val = 17
        elif row['ord_vehicleHP'] == 999:
            val = 18
        else:
            val=row['ord_vehicleHP']
        return val
    df['ord_vehicleHP'] = df.apply (recode2, axis=1)
    #Scaling
    col_names = ['num_vehicleAge', 'num_noClaimDiscountPercent','num_populationDensitykmsq',
            'num_driverAge_num','num_exposure_num']
    scaled_df = df.copy()
    features = scaled_df[col_names]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    scaled_df[col_names] = features
    #selecting columns
    scaled_df=scaled_df[['policy_desc','cat_areacode','num_vehicleAge','num_noClaimDiscountPercent',
                    'cat_carBrand','num_populationDensitykmsq','cat_Region','ord_vehicleHP','cat_fuelType',
                    'num_exposure_num','num_driverAge_num','target']].copy()
    #new dataframe with binary dummies
    enc_df=scaled_df.copy()
    enc_df=pd.get_dummies(enc_df, columns=['cat_areacode', 'cat_carBrand','cat_Region',
                'ord_vehicleHP','cat_fuelType'])
    return enc_df


def knn_model():
    """This function trains the model
    and prints the evaluation
    """
    #load the dataframe
    enc_df=data_cleaning()
    #splittin 75%/25% train-test
    training_set, test_set = train_test_split(enc_df, test_size = 0.25, random_state = 42,
                                          stratify=enc_df['target'])
    X_train = training_set.drop('target', axis=1).values
    Y_train = training_set['target'].values
    X_test = test_set.drop('target', axis=1).values
    Y_test = test_set['target'].values
    classifier=KNeighborsClassifier(n_neighbors = neighbors, algorithm='auto')
    classifier.fit(X_train,Y_train)
    Y_pred = classifier.predict(X_test)
    test_set["Predictions"] = Y_pred
    cm = confusion_matrix(Y_test,Y_pred)
    print("Confusion Matrix")
    print("****************")
    print(cm)
    accuracy = float(cm.diagonal().sum())/len(Y_test)
    print("\nAccuracy Of KNN : ", accuracy)
    return classifier

def simulator_set():
    """Returns dataframe to run simulator function
    """
    enc_df=data_cleaning()
    classifier=knn_model()
    X = enc_df.drop('target', axis=1).values
    Y = enc_df['target'].values
    predictions = classifier.predict(X)
    enc_df["Predictions"] = predictions
    result=enc_df.to_csv('simulator_set.csv', index=False)
    return result

def simulator():
    """Function that returns the number of claims predicted
    and if this prediction is right
    """
    #load dataframe
    enc_df=pd.read_csv('./simulator_set.csv',  sep =',', low_memory=False)
    policy_id=input("Please input the policy identifier (input q to quit): ")
    if policy_id=="q":
        exit()
    policy_id=int(policy_id)
    target_value=enc_df[enc_df['policy_desc']==policy_id].target.values
    prediction_value=enc_df[enc_df['policy_desc']==policy_id].Predictions.values

    if target_value==prediction_value:
        print("I say the claims for the policy {}".format(policy_id),
              "are goint to be in group {}".format(prediction_value),
             "and this is a correct prediction!")
    else:
         print("I say the claims for the policy {}".format(policy_id),
              "are goint to be {}".format(prediction_value),
             "but...oops...this is wrong, they will be in group {}".format(target_value))

    print ("\nGroups description: \n")
    print ("Group 0 -> 0 claims")
    print ("Group 1 -> 1 claim")
    print ("Group 2 -> 2 or more claims")
    simulator()

def main():
    simulator_set()
    simulator()


if __name__=='__main__':
    main()

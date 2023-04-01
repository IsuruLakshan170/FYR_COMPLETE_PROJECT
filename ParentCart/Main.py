#project initilaization
#libraries
from sklearn.model_selection import train_test_split
import pandas as pd
#files
import modelGenerator as mg
import modelTraining as mt
import modelAccuracy as ma
import dataSetSplit as sp
import modelAggregation 
import fileHandle as fh
import csv

#cart initialisation remove files that have alredy having
def resetProject():
    fh.resetModelData()


#remove stored data in carData file
def recodeDataRemove():
    

    with open('dataset/cartData.csv', 'r') as input_file:
        reader = csv.reader(input_file)
        rows = [row for row in reader]

    with open('dataset/cartData.csv', 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerows(rows[0:1])
        writer.writerows(rows[4:])

    print("Removes training data")
    


#Globle aggregation process
def globleAggregationProcess():
          print("Strat local training ------->")
          model=mg.create_model()
          model.load_weights('modelData/model_weights.h5')
          #traing model using cartdata
          print("Split dataset")
          x_train,y_train = sp.splitCartData()
          mt.continuoustrainModel(model,x_train,y_train)
          #test model using local data
          x_train_np, y_train_np,x_test_np,y_test_np =sp.splitDataset()
          ma.getModelAccuracy(model,x_test_np,y_test_np)
          #clear the csv file
          recodeDataRemove()
          #aggregate the models
          modelAggregation.modelAggregation()
          #remove received files
          fh.removeFiles()
          return "Aggregated"

#initial aggregation process  
def initialAggregationProcess():
     modelAggregation.initialModelAggregation()
     fh.removeFiles()


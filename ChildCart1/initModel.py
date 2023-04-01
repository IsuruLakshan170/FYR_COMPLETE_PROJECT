#initialization of ML mode , taing , test and save
import dataSetGenerator as ds
import modelGenerator as mg
import modelTraining as mt
import modelAccuracy as ma
import dataSetSplit as sp
import modelAggregation 
import fileHandle as fh
import client


def intModel():
    client.clientConnect()
    modelAggregation.initialModelAggregation()
    fh.removeFiles()
    
    

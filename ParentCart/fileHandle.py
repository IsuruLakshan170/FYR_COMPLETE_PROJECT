import os
import modelGenerator as mg
import saveModelData as sm

#remove the file from the initModelParameters
def removeFiles():
    directory = "receivedModelParameter" #replace with your directory path
    num_files = len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
    for i in range(num_files):
        num=i+1
        path = f'receivedModelParameter/model_weights_{num}.h5'
        try:
             os.remove(path)
        except FileNotFoundError:
             print("That file does not exist")
    print("Model parameters are removed from receivedModelParameter folder ")


def removeFilesFromBackup():
     #remove model weights
     path = f'backup/model_weights.h5'
     try:
        os.remove(path)
     except FileNotFoundError:
        print("That file does not exist")
     #remove model
     path = f'backup/model.h5'
     try:
        os.remove(path)
     except FileNotFoundError:
        print("That file does not exist")
     #remove mobile version model
     path = f'backup/model.tflite'
     try:
        os.remove(path)
     except FileNotFoundError:
        print("That file does not exist")
     print("All files are removed from backup")
     
     
def resetModelData():
    model =mg.create_model()
    sm.saveModelData(model)
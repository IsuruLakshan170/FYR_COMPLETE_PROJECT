#get model accuracy

#libraries
from sklearn.metrics import accuracy_score
import numpy as np
import modelGenerator as mg

def getModelAccuracy(model,test_data1,test_labels1):
      #Predict model 1  test using test date
    y_pred_model_1 = model.predict(test_data1)

    # The predictions are in the form of probability of each class, so we will take the class with highest probability
    y_pred_model_1 = y_pred_model_1.argmax(axis=-1)

    # Calculate the accuracy of the model
    
    accuracy_model_1 = accuracy_score(test_labels1, y_pred_model_1)
    print("Model  Accuracy:", accuracy_model_1*100)
    return accuracy_model_1
  
def predictionsResults(model,test_data1):
  #Predict model 1  test using test date
  y_pred_model_1 = model.predict(test_data1)

  # The predictions are in the form of probability of each class, so we will take the class with highest probability
  y_pred_model_1 = y_pred_model_1.argmax(axis=-1)

  # Calculate the accuracy of the model
  # print("Predicted Results")
  # print(y_pred_model_1)
  return y_pred_model_1
  
def getCurrentThreand(month,gender):
  x_data=[month,gender] 
  y_data=[1] 

  x_np = np.array(x_data)
  x_np = x_np.reshape(1, 2)
  x_np = x_np.astype('float32')
  x_np /= 12

  y_np = np.array(y_data)
  y_np = y_np.astype('float32')
  # print(type(x_np))
  # print((x_np))
  # print(type(y_np))
  model = importModel()
  results = predictionsResults(model,x_np)
  return results[0]
 
  
def importModel():
   model=mg.create_model()
   model.load_weights('modelData/model_weights.h5')
   return model

# results = getCurrentThreand(1,0)
# print(results)
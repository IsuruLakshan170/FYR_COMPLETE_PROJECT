#train the ML model
#libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping

#import files
import saveModelData as sm

def continuoustrainModel(model,train_data1,train_labels1):
    # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    model.fit(train_data1, train_labels1, epochs=1, batch_size=128, validation_split=0.2, callbacks=[early_stopping])
    #save model data
    sm.saveModelData(model)
    print("Localy trained and saved model parameters")
    return model
   

   
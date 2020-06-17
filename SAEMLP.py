import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

data = pd.read_csv('dataset_sdn.csv')

for i in range(len(data)):
    if(data['pktrate'][i] < 0):
        data.drop(i,axis = 0,inplace = True)

data = data.dropna()
np.count_nonzero(data['label'])

data = data.iloc[:,1:]

count = 0
random = np.random.randint(0,len(data),(50000))

for i in random:
    
    if(count == 10000):
        break
    try:
        index = data['label'][i]
    except:
        continue
    else:   
        if(data['label'][i] == 0):
            data.drop(i,axis = 0,inplace = True)
            count += 1

np.count_nonzero(data['label'])

data['switch'] = data['switch'].astype(str) 
data['src'] = data['src'].astype(str)
data['dst'] = data['dst'].astype(str)
data['port_no'] = data['port_no'].astype(str)
data['Protocol'] = data['Protocol'].astype(str)

ordered_data = pd.get_dummies(data,
                              columns = ['switch','src','dst','Protocol','port_no'])
#ordered_data.head()
columns = np.array(ordered_data.columns)

#columns_dict = {string:i for i,string in enumerate(columns)}
values = ordered_data.values

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(values[:,:16])
scaled_values = scaler.transform(values[:,:16])

new_values = np.hstack([values[:,17:26],values[:,27:45],scaled_values,
                        values[:,46:62],values[:,63:65],values[:,66:70]])

labels = values[:,16]

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(new_values,
                                                   labels,test_size = 0.1)    
X_test,X_valid,Y_test,Y_valid = train_test_split(X_test,Y_test,
                                                 test_size = 0.5)
    
from keras.layers import Input,Dense 
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

begin_time = time.time()
dense_1 = Dense(25,activation = 'tanh')
dense_2 = Dense(8,activation = 'selu')
dense_3 = Dense(25,activation = 'tanh')

inputs = Input(shape = (65,))
dense_x1 = dense_1(inputs)
dense_x2 = dense_2(dense_x1)
dense_x3 = dense_3(dense_x2)
outputs = Dense(65,activation = 'linear',name = 'reconstruction')(dense_x3)

model = Model(inputs = inputs,outputs = outputs)

#dimensionality reduction
optimizer = SGD(learning_rate = 1.2,momentum = 0.8)
        
model.compile(optimizer = optimizer,
                 loss = 'mse',metrics = ['accuracy'])
history_0 = model.fit(X_train,X_train,validation_data = [X_valid,X_valid],
          epochs = 10,batch_size = 400) 

filepath = "best_weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', 
                             verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.compile(optimizer = 'adam',loss = 'mse',metrics = ['accuracy'],)
model.summary()

history_1 = model.fit(X_train,X_train,validation_data = [X_valid,X_valid],
          epochs = 150,batch_size = 400,callbacks = callbacks_list) 

#summarize history on Reconstruction Accuracy

x_0 = np.linspace(0,9,10,dtype = np.int32)
x_1 = np.linspace(10,159,150,dtype = np.int32)

training_accuracy_0 = np.array(history_0.history['accuracy'])
validation_accuracy_0 = np.array(history_0.history['val_accuracy'])
plt.plot(x_0,training_accuracy_0[x_0],color = 'blue',label = 'Train')
plt.plot(x_0,validation_accuracy_0[x_0],color = 'orange')

training_accuracy_1 = np.array(history_1.history['accuracy'])
validation_accuracy_1 = np.array(history_1.history['val_accuracy'])
plt.plot(x_1,training_accuracy_1[x_1-10], color = 'blue')
plt.plot(x_1,validation_accuracy_1[x_1-10], color = 'orange',label = 'Valid')

plt.xlim(-10,170)
plt.ylim(0.45,0.8)

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('SAE Reconstruction Accuracy')
plt.legend(frameon = True,loc = 'lower right')

#summarize history on Reconstruction Loss

training_loss_0 = np.array(history_0.history['loss'])
validation_loss_0 = np.array(history_0.history['val_loss'])
plt.plot(x_0,training_loss_0[x_0],color = 'blue',label = 'Train')
plt.plot(x_0,validation_loss_0[x_0],color = 'orange')

training_loss_1 = np.array(history_1.history['loss'])
validation_loss_1 = np.array(history_1.history['val_loss'])
plt.plot(x_1,training_loss_1[x_1-10], color = 'blue')
plt.plot(x_1,validation_loss_1[x_1-10], color = 'orange',label = 'Valid')

plt.xlim(-10,170)

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('SAE Reconstruction Loss')
plt.legend(frameon = True,loc = 'upper right')

#classification
model.load_weights(filepath)
model.layers.pop()
model.layers.pop()

output_2 = Dense(1,activation = 'sigmoid')(model.layers[-1].output)
model_2 = Model(inputs = model.inputs , outputs = output_2)
model_2.compile(optimizer = 'adam',loss = 'binary_crossentropy',
                metrics = ['accuracy'])
model_2.summary()
history_2 = model_2.fit(X_train,Y_train,validation_data = [X_test,Y_test],
            epochs = 50,batch_size = 400)

end_time = time.time()

#summarize history on classification Accuracy

x_2 = np.linspace(0,49,50,dtype = np.int32)

training_accuracy_2 = np.array(history_2.history['accuracy'])
validation_accuracy_2 = np.array(history_2.history['val_accuracy'])

plt.plot(x_2,training_accuracy_2[x_2],color = 'blue',label = 'Train')
plt.plot(x_2,validation_accuracy_2[x_2],color = 'orange',label = 'Valid')

plt.xlim(-2,52)
plt.ylim(0.80,1.05)

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('SAE_MLP Classification Accuracy')
plt.legend(frameon = True,loc = 'lower right')

#summarize history on classification loss

training_loss_2 = np.array(history_2.history['loss'])
validation_loss_2 = np.array(history_2.history['val_loss'])

plt.plot(x_2,training_loss_2[x_2],color = 'blue',label = 'Train')
plt.plot(x_2,validation_loss_2[x_2],color = 'orange',label = 'Valid')

plt.xlim(-2,52)
plt.ylim(-0.1,0.6)

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('SAE_MLP Classification Loss')
plt.legend(frameon = True,loc = 'upper right')


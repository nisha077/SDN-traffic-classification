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

X_train = X_train.reshape(len(X_train),65,1)
X_valid = X_valid.reshape(len(X_valid),65,1)
X_test = X_test.reshape(len(X_test),65,1)

from keras.layers import Input
from keras.layers import Conv1D,MaxPooling1D
from keras.layers import LSTM,Dense
from keras.models import Model

begin_time = time.time()

inputs = Input(shape = (65,1,))
x = Conv1D(filters = 64,kernel_size = 3,padding = 'same',activation = 'relu')(inputs)
x = MaxPooling1D(pool_size = 2,padding = 'same')(x)
x = LSTM(128,return_sequences = True)(x)
x = LSTM(128)(x)
outputs = Dense(1,activation ='sigmoid')(x)

model = Model(inputs = inputs,outputs = outputs)
model.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

model.summary()

from keras.callbacks import ModelCheckpoint

filepath = "best_weights_CNN_LSTM.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', 
                             verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit(X_train,Y_train,validation_data = [X_valid,Y_valid],
          epochs = 10,batch_size = 64,callbacks = callbacks_list)

model.load_weights(filepath)
model.evaluate(X_test,Y_test)

end_time = time.time()

#summarize history on classification Accuracy

x = np.linspace(0,9,10,dtype = np.int32)

training_accuracy = np.array(history.history['accuracy'])
validation_accuracy = np.array(history.history['val_accuracy'])

plt.plot(x,training_accuracy[x],color = 'blue',label = 'Train')
plt.plot(x,validation_accuracy[x],color = 'orange',label = 'Valid')

plt.xlim(-1,11)
plt.ylim(0.70,1.0)

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('CNN_LSTM Classification Accuracy')
plt.legend(frameon = True,loc = 'lower right')

#summarize history on classification loss

training_loss = np.array(history.history['loss'])
validation_loss = np.array(history.history['val_loss'])

plt.plot(x,training_loss[x],color = 'blue',label = 'Train')
plt.plot(x,validation_loss[x],color = 'orange',label = 'Valid')

plt.xlim(-1,11)
plt.ylim(-0.1,0.45)

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('CNN_LSTM Classification Loss')
plt.legend(frameon = True,loc = 'upper right')

predict = model.predict(X_test)
predict= np.round(predict)
predict = [int(x) for x in predict]
from sklearn.metrics import confusion_matrix

tp_ls,fp_ls,fn_ls,tn_ls = confusion_matrix(Y_test, predict).ravel()
acc_ls=(tp_ls+tn_ls)/(tp_ls+fp_ls+fn_ls+tn_ls)
rec_ls=tp_ls/(tp_ls+fn_ls)
rec_lsa=tn_ls/(tn_ls+fp_ls)
pre_ls=tp_ls/(tp_ls+fp_ls)
pre_lsa=tn_ls/(tn_ls+fn_ls)
fpr_ls=fp_ls/(fp_ls+tn_ls)
fnr_ls=fn_ls/(fn_ls+tp_ls)
f_ls=2*pre_ls*rec_ls/(pre_ls+rec_ls)
f_lsa=2*pre_lsa*rec_lsa/(pre_lsa+rec_lsa)

import matplotlib.pyplot as plt
normal7 = {}
#normal['Accuracy'] = acc_ls
normal7['Precision'] = pre_ls
normal7['Recall'] = rec_ls
normal7['F-Score'] = f_ls

labels = ['Accuracy','Precision','Recall','F-Score']

attack7 = {}
#attack['Accuracy'] = acc_l
attack7['Precision'] = pre_lsa
attack7['Recall'] = rec_lsa
attack7['F-Score'] = f_lsa


index= np.arange(1,4)
yoo=np.arange(4)

height01= list(normal7.values())
height02= list(attack7.values())

bar_width = 0.35
plt.bar(0.35/2, acc_ls, label='Accuracy', width=bar_width)
plt.bar(index, height01, label='normal', width=bar_width)
plt.bar(index+bar_width, height02, label='attack', width=bar_width)
plt.xticks(yoo+bar_width/2, labels)
plt.title('CNN_LSTM')
plt.legend()
plt.show()

print(acc_ls, pre_ls, pre_lsa, rec_ls, rec_lsa, f_ls, f_lsa, fpr_ls, fnr_ls,sep = ' ')
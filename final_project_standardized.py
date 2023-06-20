#pip install torch, transformers, datasets, tensorflow, flax, keras before running
#add code citations

#import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf
import matplotlib.pyplot as plt

#import tokenizer and pre-trained model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
#switched from BertTokenizer and TFBertForSequenceClassification to DistilBert...
#... because it was taking hours locally (eventually crashed my computer)...
#... and maxing out free Google Colab RAM. Now it takes about 30 minutes locally and 2 hours in Colab.

#random seed
random_seed = 42

#set random seed in tensorflow
tf.random.set_seed(random_seed)

#set random seed in numpy
np.random.seed(random_seed)

#load the raw training data
df_raw_train = pd.read_csv("data/train.csv")
#make a copy of df_raw_train
df_train = df_raw_train.copy(deep=True)

#load the raw test data
df_raw_test = pd.read_csv("data/test.csv")
#make a copy of df_raw_test
df_test = df_raw_test.copy(deep=True)

#get target name
target = 'target'

#print shape of df_train
pd.DataFrame([[df_train.shape[0], df_train.shape[1]]], columns=['# rows', '# columns'])

#print shape of df_test
pd.DataFrame([[df_test.shape[0], df_test.shape[1]]], columns=['# rows', '# columns'])

#print head of df_train
df_train.head()

#print head of df_test
df_test.head()

#drop columns
df_train.drop(['id','keyword','location'],axis=1,inplace=True)
df_test.drop(['id','keyword','location'],axis=1,inplace=True)

#print head of df_train to ensure dropping worked correctly
df_train.head()

#print head of df_test to ensure dropping worked correctly
df_test.head()

#batch tokenize our tweet field
X_train = tokenizer.batch_encode_plus(df_train.text, pad_to_max_length=True, return_tensors="tf")
X_test = tokenizer.batch_encode_plus(df_test.text, pad_to_max_length=True, return_tensors="tf")

#obtain target
y_train = df_train['target'].to_numpy()

#optimize model
#fine tune parameters? https://github.com/uzaymacar/comparatively-finetuning-bert
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
#try AdamW? https://towardsdatascience.com/why-adamw-matters-736223f31b5d
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
bce = tf.keras.losses.BinaryCrossentropy()
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

#compile and train model on training data
#https://stackoverflow.com/questions/46308374/what-is-validation-data-used-for-in-a-keras-sequential-model
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
history = model.fit(x=X_train['input_ids'], y=y_train, epochs=2, batch_size=15, verbose=2, validation_split=0.2)
#previously had more epochs, but reduced iterations as per official guidance from BERT's documentation...
#... (and also it was taking forever)

#evaluate
#add plot to show val_loss compared to train loss

history.history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show

#add plot to show val_accuracy compared to train accuracy.
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

#predict
predictions = model.predict(X_test['input_ids'])
predictions_label = [np.argmax(x) for x in predictions[0]]
#print results
results = pd.DataFrame({'id': df_raw_test['id'], 'target': predictions_label})
results['target'] = results['target'].astype('int')
results.to_csv('predictions.csv', index=False)
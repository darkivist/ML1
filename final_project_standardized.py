#pip install torch, transformers, datasets, tensorflow, flax, keras, sklearn before running
#add code citations

#import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf

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

#training (80%) and validation (20%) data split

df_train, df_val = train_test_split(df_train, train_size=0.8, random_state=random_seed)

#reset index
df_train, df_val = df_train.reset_index(drop=True), df_val.reset_index(drop=True)

#print shape of df_train
pd.DataFrame([[df_train.shape[0], df_train.shape[1]]], columns=['# rows', '# columns'])

#print shape of df_val
pd.DataFrame([[df_val.shape[0], df_val.shape[1]]], columns=['# rows', '# columns'])

#batch tokenize our tweet field
X_train = tokenizer.batch_encode_plus(df_train.text, pad_to_max_length=True, return_tensors="tf")
X_val = tokenizer.batch_encode_plus(df_val.text, pad_to_max_length=True, return_tensors="tf")
X_test = tokenizer.batch_encode_plus(df_test.text, pad_to_max_length=True, return_tensors="tf")

#obtain target
y_train = df_train['target'].to_numpy()
y_val = df_val['target'].to_numpy()

#optimize model
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
bce = tf.keras.losses.BinaryCrossentropy()
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

#compile and train model on training data
#do we need to add random seed to model.fit or am I misremembering?
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
model.fit(x=X_train['input_ids'], y=y_train, epochs=2, validation_data=(X_val,y_val), batch_size=15, verbose=1)
#epochs 2, batch size 15 resulted in loss: 0.3146 - accuracy: 0.8760
#previously had more epochs, but reduced iterations as per official guidance from BERT's documentation...
#... (and also it was taking forever)
#fine tune parameters? https://github.com/uzaymacar/comparatively-finetuning-bert

#compile and train model on validation data - how can we take what was learned in the initial training and feed it into here?
#model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
#model.fit(x=X_val['input_ids'], y=y_val, epochs=2, batch_size=15, verbose=1)

#predict
predictions = model.predict(X_test['input_ids'])
predictions_label = [np.argmax(x) for x in predictions[0]]
#print results
results = pd.DataFrame({'id': df_raw_test['id'], 'target': predictions_label})
results['target'] = results['target'].astype('int')
results.to_csv('predictions.csv', index=False)
#profit!
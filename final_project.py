#before doing this, pip install torch, transformers, datasets, tensorflow, flax, keras, sklearn
#add code citations

#importing libraries and tokenizers
#tokenizer: https://www.techtarget.com/searchenterpriseai/definition/BERT-language-model
import pandas as pd
from transformers import BertTokenizer, DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf
import numpy as np

#switched from BertTokenizer and TFBertForSequenceClassification to DistilBert...
#... because it was taking hours locally (eventually crashed my computer)...
#... and maxing out free Google Colab RAM. Now it takes about 30 minutes.

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

#run the model on cpu.
#model.cpu()

#read in data
training_orig = pd.read_csv("data/train.csv")
#create working copy
training_work=training_orig.copy()
#ensure columns imported correctly
print(training_work.columns)
#put some more EDA here
#print dataframe head
print(training_work.head(10))
#drop columns
training_work.drop(['id','keyword','location'],axis=1,inplace=True)
#set tweet and disaster_value variables
tweets = training_work.text.values
disaster_value = training_work.target.values
#print culled dataset head to ensure dropping worked correctly
print(training_work.head())
#test tokenizer
#print the original tweet example
print(' original: ', tweets[0])
#print tokenized tweet as python list
print('tokenized: ', tokenizer.tokenize(tweets[0]))
#print tweet mapped to tokenized tweet
print('token id: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tweets[0])))
#the above shows we either need to use a tokenizer that gets rid of hashtags, etc...
# ...or we need to do more preprocessing before tokenizing...
#... OR do we not drop '#' because a hashtag might indicate a disaster event?
#train/val split

#preprocessing
train_x = tokenizer.batch_encode_plus(tweets, pad_to_max_length=True, return_tensors="tf")
train_y = training_work['target'].to_numpy()
#train_y = disaster_values.to_numpy()
print(train_x)
#optimize model
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
bce = tf.keras.losses.BinaryCrossentropy()
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
#compile and train model on training data
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
model.fit(x=train_x['input_ids'], y=train_y, epochs=2, batch_size=15, verbose=1)
#do we need to add random seed to model.fit or am I misremembering?
#epochs 2, batch size 15 resulted in loss: 0.3146 - accuracy: 0.8760
#previously had more epochs, but reduced iterations as per official guidance from BERT's documentation...
#... (and also it was taking forever)
#fine tune parameters? https://github.com/uzaymacar/comparatively-finetuning-bert

#should we split train data into train and valid with sklearn?

#read in test data
test_orig = pd.read_csv("data/test.csv")
#create working copy of test data
test_work=test_orig.copy()
#repeat tokenization
test_x = test_work['text'].to_numpy()
test_x = tokenizer.batch_encode_plus(test_x, pad_to_max_length=True, return_tensors="tf")
#predict
predictions = model.predict(test_x['input_ids'])
predictions_label = [np.argmax(x) for x in predictions[0]]
#print results
results = pd.DataFrame({'id': test_work['id'], 'target': predictions_label})
results['target'] = results['target'].astype('int')
results.to_csv('predictions.csv', index=False)
#profit!
#before doing this, pip install torch, transformers, datasets, tensorflow, flax, keras, sklearn

#importing libraries and tokenizers
#tokenizer: https://www.techtarget.com/searchenterpriseai/definition/BERT-language-model
import pandas as pd
from transformers import BertTokenizer, DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf

#switched from BertTokenizer and TFBertForSequenceClassification to DistilBert because it was taking hours locally and maxing out Google Colab RAM

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
#print dataframe head
print(training_work.head(10))
#drop columns
training_work.drop(['id','keyword','location'],axis=1,inplace=True)
#set tweet and disaster_value variables
tweets = training_work.text.values
disaster_value = training_work.target.values
#print culled dataset head to ensure dropping worked correctly
print(training_work.head())
#print the original tweet example
print(' original: ', tweets[0])
#print tokenized tweet as python list
print('tokenized: ', tokenizer.tokenize(tweets[0]))
#print tweet mapped to tokenized tweet
print('token id: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tweets[0])))
#the above shows we either need to use a tokenizer that gets rid of hashtags, etc., or we need to do more preprocessing before tokenizing

train_x = tokenizer.batch_encode_plus(tweets, pad_to_max_length=True, return_tensors="tf")
train_y = training_work['target'].to_numpy()
#train_y = disaster_values.to_numpy()
print(train_x)

#optimize

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
bce = tf.keras.losses.BinaryCrossentropy()
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
#previously had more epochs, but reduced iterations as per official guidance from BERT's documentation and also it was taking forever
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
model.fit(x=train_x['input_ids'], y=train_y, epochs=2, batch_size=15, verbose=1)
#this resulted in loss: 0.3146 - accuracy: 0.8760

#next - maybe amend the above to utilize a train/validation split, re-run model
#next next - read in test data, prep as with train, run prediction
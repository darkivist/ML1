#before doing this, pip install torch, transformers, datasets, tensorflow, flax, keras, sklearn

#importing libraries and tokenizers
#tokenizer: https://www.techtarget.com/searchenterpriseai/definition/BERT-language-model
import pandas as pd
from transformers import BertTokenizer
import tensorflow as tf
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
from transformers import TFBertForSequenceClassification


model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")

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

model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
model.fit(x=train_x['input_ids'], y=train_y, epochs=15, batch_size=128, verbose=1)
#before doing this, pip install torch, transformers, datasets, tensorflow, flax

#importing libraries and tokenizers
#tokenizer: https://www.techtarget.com/searchenterpriseai/definition/BERT-language-model
import pandas as pd
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

#read in data
training_orig = pd.read_csv("data/train.csv")
#backup data
training_work=training_orig.copy()
#ensure columns imported correctly
print(training_work.columns)
#print dataframe head
print(training_work.head(10))
#drop columns
training_work.drop(['id','keyword','location'],axis=1,inplace=True)
#set tweet and disaster_value variables
tweet = training_work.text.values
disaster_value = training_work.target.values
#print culled dataset head to ensure dropping worked correctly
print(training_work.head())
#print the original tweet example
print(' original: ', tweet[0])
#print tokenized tweet as python list
print('tokenized: ', tokenizer.tokenize(tweet[0]))
#print tweet mapped to tokenized tweet
print('token id: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tweet[0])))
#the above shows we either need to use a tokenizer that gets rid of hashtags, etc., or we need to do more preprocessing before tokenizing
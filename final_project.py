#before doing this, pip install torch, transformers, datasets, tensorflow, flax, keras

#importing libraries and tokenizers
#tokenizer: https://www.techtarget.com/searchenterpriseai/definition/BERT-language-model
import pandas as pd
from transformers import BertTokenizer
import keras
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

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

#put data into correct shape for BERT
input_ids = []
for tweet in tweets:
    # so basically encode tokenizing , mapping sentences to thier token ids after adding special tokens.
    encoded_sent = tokenizer.encode(
        tweet,  # tweet which are encoding.
        add_special_tokens=True,  # Adding special tokens '[CLS]' and '[SEP]'

    )

    input_ids.append(encoded_sent)


#pad/truncate

from keras.utils import pad_sequences
MAX_LEN = 128
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN , truncating="post", padding="post")

#ensure model focuses on data and not padding

attention_masks = []

for tweet in input_ids:
    # Generating attention mask for sentences.
    #   - when there is 0 present as token id we are going to set mask as 0.
    #   - we are going to set mask 1 for all non-zero positive input id.
    att_mask = [int(token_id > 0) for token_id in tweet]

    attention_masks.append(att_mask)

#training/validation split

from sklearn.model_selection import train_test_split

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, disaster_value, test_size=0.2)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, disaster_value,test_size=0.1)



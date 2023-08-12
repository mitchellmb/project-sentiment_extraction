'''
Process textual data to create sentiment values & score 
    - using pretrained roberta model from https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment
    - fine tuning bert-base-uncased model from review_text_fine_tune.py and https://huggingface.co/bert-base-uncased
    - fit time ~ 15 minutes / 100K reviews
'''
#%%
import numpy as np
import re
import torch
from nltk.tokenize import sent_tokenize

class TextPrepare():
    ''' 
    Text processing class to extract sentence length, text length, and sentiments from bert based language models.
    
    Attributes
    ----------
    text : str
        review text
        
    length : int
        count of characters in text
        
    sentence_count : int
        count of sentences in text
        
    split : str
        truncated text up to max_length
        
    sentiment_value : float
        roberta pretrained sentiment (-1, 1) of text
        
    score : float
        roberta pretrained score (0, 1) of sentiment_value
        
    ft_positive : float
        bert fine tuned positive score (0, 1) of text
        
    ft_neutral : float
        bert fine tuned neutral score (0, 1) of text
        
    ft_negative : float
        bert fine tuned negative score (0, 1) of text
        
        
        
    Methods
    -------
    len():
        returns length of text
        
    strip():
        removes new line characters and url's from text
        
    split(max_length):
        returns truncated text up to max_length
        
    roberta_fit(pipe, sentiment_dictionary, max_length):
        returns sentiment_value and score for pretrained roberta sentiment analysis
        
    ft_bert_fit(tokenizer, model):
        returns positive, neutral, and negative sentiments for fine tuned bert model
    '''
    
    def __init__(self, text):
        #Initializes the necessary attributes for the TextPrepare object.
        self.text = text
        self.length = len(text)
        self.sentence_count = len(sent_tokenize(text))
        self.split = None
        
        self.sentiment_value = None
        self.score = None
        self.ft_positive = None
        self.ft_neutral = None
        self.ft_negative = None
        
    def __len__(self):
        return self.length 
        
    def strip(self):
        #Remove \n characters
        self.text = self.text.replace('\n', '')
        #Remove url's
        self.text = re.sub(r'http\S+', ' ', self.text)
        
    def split(self, max_length):
        self.split = [self.text[x:x+max_length] for x in range(0, len(self.text), max_length)]
        
    def roberta_fit(self, pipe, sentiment_dictionary, max_length):
        #Fit pretrained roberta and extract sentiment & sentiment score
        if self.length < max_length:
            sentiment, self.score = pipe(self.text)[0].values()
            self.sentiment_value = sentiment_dictionary[sentiment]
            
        else:
            #Split the long text up into sections defined by max_length
            split = [self.text[x:x+max_length] for x in range(0, len(self.text), max_length)]
            split_piped = [list(pipe(split[s])[0].values()) for s in range(0, len(split))]
            scores = [item[1] for item in split_piped]
            sentiments = [item[0] for item in split_piped]
            
            #Average sentiment values and scores to produce a single overall value for the class instance
            self.sentiment_value = np.average(
                [sentiment_dictionary[sentiments[i]] for i in range(0,len(sentiments))])
            
            self.score = np.average(scores)
    
    def ft_bert_fit(self, tokenizer, model):
        #Fit fine tuned bert model and extract sentiment scores for positive, neutral, and negative
        inputs = tokenizer(self.text, padding=True, truncation=True, return_tensors='pt').to('cuda')
        outputs = model(**inputs)
        prediction = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().detach().numpy()[0]
        
        self.ft_positive = prediction[2]
        self.ft_neutral = prediction[1]
        self.ft_negative = prediction[0]
            
            
                
            
                
        


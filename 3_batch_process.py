"""
Batch processing of all 6.9 million reviews in 100,000 chunks
    - Follows steps made during data exploration: 0_Yelp_reviews_data_exploration.py
      but in an interable fashion
    - These data will be analyzed in the 4_XGBoost_model.ipynb notebook
"""
import numpy as np
import pandas as pd
import joblib
import time
from datetime import timedelta
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from text_preprocess import TextPrepare

#%%
#Prepare dependencies for each chunk processing:

#--------------------------------------------------------------------------------------------------------#    
#1: Text analysis

#Pretrained roBERTa model pipeline and utilize cuda computing with device=0
model_roberta = 'cardiffnlp/twitter-xlm-roberta-base-sentiment'
pipe_roberta = pipeline('sentiment-analysis', 
                        model=model_roberta, 
                        tokenizer=model_roberta, 
                        device_map='auto') #auto uses Accelerate to choose cpu/gpu map

#Finetuned bert-base-uncased model on ~0.07% of reviews
path = r'C:\Users\mitch\.cache\huggingface\hub'
model_finetuned_name = path + r'\models--bert-base-uncased-YR-finetuned'
model_finetuned = AutoModelForSequenceClassification.from_pretrained(model_finetuned_name, num_labels=3)
model_tokens = 'bert-base-uncased'
tokenizer_finetuned = AutoTokenizer.from_pretrained(model_tokens)
config = AutoConfig.from_pretrained(model_finetuned_name)
model_finetuned.to('cuda')

sentiment_dictionary = {'positive': 2, 'neutral': 1, 'negative': 0} #encoded sentiments
max_text_length = 1000 #splits into multiple text sections, each of which are evaluated

text_col_names = ['length', 'sentence count', #generic text info
                 'pt sentiment', 'pt sentiment score', #pretrained roBERTa output
                 'ft positive', 'ft neutral', 'ft negative'] #finetuned BERT output


#%%
#--------------------------------------------------------------------------------------------------------#   
#2: Business type

#7 subset category groups
food = ['food', ['food','restaurants']]

shop = ['shop', ['shopping','automotive','pets']]

housing = ['housing',['real estate']]

education = ['education', ['education']]

entertainment = ['entertainment',['arts & entertainment','beauty & spas','nightlife',
                 'local flavor','mass media','active life','hotels & travel','tours']]

health = ['health',['health & medical']]

services = ['services', ['event planning & services','financial services','car rental',
            'home services','local services','professional services','public services & government',
            'religious organizations']]

category = [food, shop, housing, education, entertainment, health, services]
business_type_col_names = ['food','shop','health','housing','education','entertainment','services']

def primary_business_type(row):
    #find first nonzero instance index
    primary = np.where(row == 1)[0][0]
    #create array to set non-primary keys to 0
    mask = np.zeros(len(row))
    mask[primary] = 1
    return mask*row

#%%
#--------------------------------------------------------------------------------------------------------#   
#3: K-means clustering by latitude/longitude

#Re-use the kmeans clustering fit from Yelp_reviews_data_exploration.ipynb
kmeans_pretrained = joblib.load(r'../ML_models/kmeans_latlong.joblib')
                                
#%%
#--------------------------------------------------------------------------------------------------------#   
def process_chunk(chunk, ID):
    '''
    Iterable function over a chunk with chunk number = ID.
    
    Analysis:
        1: Text analysis returns 
            - roBERTa pretrained sentiment model analysis
            - bert-base-uncased fine tuned model analysis
            - text length + sentence count
            
        2: Business type returns
            - corresponding generalized business type for a review (7 categories defined above)
            
        3: Cluster returns
            - reviews grouped in 11 major clusters by latitude/longitude
        
    Returns:
        4: Combined analysis results in a new dataframe
        
        5: Writes dataframe to csv according to chunk ID label
    '''
    
    #1: Text analysis
    text_input = chunk['text']
    
    text_information = []
    for i in range(0, len(text_input)):
        
        text_info = TextPrepare(text_input[i])
        text_info.strip()
        text_info.roberta_fit(pipe_roberta, sentiment_dictionary, max_text_length)
        text_info.ft_bert_fit(tokenizer_finetuned, model_finetuned)

        text_information.append([text_info.length,
                                 text_info.sentence_count,
                                 
                                 #roBERTa pretrainted model
                                 text_info.sentiment_value, 
                                 text_info.score, 
                                
                                 #BERT fine tuned
                                 text_info.ft_positive,
                                 text_info.ft_neutral,
                                 text_info.ft_negative
                                ])
        
    text_analysis = pd.DataFrame(text_information, columns=text_col_names)
    
    
    #2: Business type
    #Extract a more generic category for each review and one-hot encode them
    
    #Initialize with a 0's dataframe of the same shape as categories
    business_type = pd.DataFrame(0, index=np.arange(len(chunk['categories'])),
                                 columns=business_type_col_names)
    
    #Count the number of instances of category indication per review
    for i in range(0, len(chunk['categories'])):
        for j in category:
            for k in j[1]:
                if k in chunk['categories'][i].lower().split(', '):
                    business_type[j[0]][i] = 1
                    
    for i in range(0, len(business_type)):
        business_type.loc[i] = primary_business_type(business_type.loc[i]) 
        
        
    #3: K-means clustering latitude/longitude
    cluster = kmeans_pretrained.predict(chunk[['latitude','longitude']])
    
    
    #4: combine results and drop unneeded columns
    chunk = chunk.merge(text_analysis, left_index=True, right_index=True)
    chunk = chunk.merge(business_type, left_index=True, right_index=True)
    chunk['cluster'] = cluster

    chunk.drop(columns=['state', 'latitude', 'longitude', 'categories'], inplace=True)
    
    #5: write the output for this batch
    chunk.to_csv(r'../csv_processed/batches/' + str(ID) + r'_batch.csv', index=False)
    
#%%
#--------------------------------------------------------------------------------------------------------#  
#Load all data points in batches and process each one sequentially
    #A 100,000 batch takes about ~22m to preprocess fully
    #70 batches to run

chunk_size = 100000
ID = 0 #last chunk completed is ID-1 if stopping midway through
 
for i in range(0, 64): 
    chunk = pd.read_csv(r'../csv_processed/0_collated_yelp_reviews.csv', 
                        skiprows = range(1, ((ID + i) * chunk_size) + 1), 
                        nrows = chunk_size, 
                        index_col = 0).reset_index(drop=True)
    
    start = time.time()
    process_chunk(chunk, ID)
    
    stop = time.time()
    duration = timedelta(seconds=(stop-start))
    print(f'Finished batch: {ID} / 70 in {duration}')
    ID += 1
    

'''
Fine tuning of a bert-base-uncased https://huggingface.co/bert-base-uncased model to 50k reviews (~0.7% of all data)
'''

#%%
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

#%% 
#Load in a subset of data
business_reviews = pd.read_csv(r'..\csv_processed\0_collated_yelp_reviews.csv', 
                               index_col=0,
                               nrows=50000) #smaller subset of data to train on ~ 0.7% of total data

#Attempt to get a more generic bad = 0, average = 1, great = 2 score
stars_map = {1:0, 2:0, 3:1, 4:2, 5:2}
business_reviews['review stars'] = business_reviews.replace({'review stars': stars_map})['review stars']

#%%
#Fine tuning the model
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import evaluate

#Setup the model, port to GPU
model = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model)
bert_model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=3) #0,1,2 for generic stars_map
config = AutoConfig.from_pretrained(model)
bert_model.to('cuda')

#Train-test split 
X = list(business_reviews['text'])
y = list(business_reviews['review stars'])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)
X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)

#%%
#Convert tokenized format output into a readable dataset for PyTorch 
    #need __init__, __getitem__, __len__

class Dataset_pt(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

#Define a metric to evaluate on in training
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

#%%
train_dataset = Dataset_pt(X_train_tokenized, y_train)
val_dataset = Dataset_pt(X_val_tokenized, y_val)

#Create trainer
training_args = TrainingArguments(output_dir = r'../ML_models/test_trainer', 
                                  num_train_epochs=2,  
                                  learning_rate=2e-5,
                                  per_device_train_batch_size=8,  
                                  per_device_eval_batch_size=16,
                                  warmup_steps=50,                
                                  weight_decay=0.01               
                                  )

trainer = Trainer(model=bert_model,
                  args=training_args,
                  train_dataset=train_dataset,
                  eval_dataset=val_dataset,
                  compute_metrics=compute_metrics)

#Train and evaluate on test/train split
trainer.train()
trainer.evaluate()  
#%%
#Test the model
text_test = 'The patio was nice.'
inputs = tokenizer(text_test, padding=True, truncation=True, return_tensors='pt').to('cuda')
outputs = bert_model(**inputs)
print(outputs)

prediction = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().detach().numpy()
print(prediction)

#%%
#Save the model to use in main text processing pipeline
trainer.save_model(r'../ML_models/models--bert-base-uncased-YR-finetuned')

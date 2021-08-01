#!/usr/bin/env python
# coding: utf-8

# In[9]:


from Prediction import Prediction
from TextCleaner import TextCleaner
from DataPreprocessing import DataPreprocessing


# In[19]:


from keras.preprocessing.sequence import pad_sequences


# In[11]:


predictor = Prediction()
cleaner = TextCleaner()
processor = DataPreprocessing()


# In[12]:


loaded_data = processor.load_pickle('TokenizerData')

x_tokenizer, y_tokenizer, x_vocab_size,y_vocab_size, input_word_index,target_word_index, reversed_input_word_index, reversed_target_word_index, max_length_text, max_length_summary = loaded_data[0],loaded_data[1], loaded_data[2],loaded_data[3],loaded_data[4],loaded_data[5],loaded_data[6],loaded_data[7],loaded_data[8],loaded_data[9]


# In[3]:


# Load trained model
encoder_model = predictor.load_model('encoder_model.json', 'encoder_model_weights.h5')
decoder_model = predictor.load_model('decoder_model.json', 'decoder_model_weights.h5')
print("Model Loaded.")


# In[4]:


# Generate summaries
predictor.generated_summaries(10, encoder_model, decoder_model)


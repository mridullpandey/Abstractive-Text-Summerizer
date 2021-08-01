#!/usr/bin/env python
# coding: utf-8

# In[3]:


class Main():
    """Data Preprocessing"""
    from DataPreprocessing import DataProprocessing
    processor = DataProprocessing()
    # Read-in dataset
    data = processor.load_dataset('cnn')
    print('Dataset Loaded.')
    
    # clean data
    data['Stories'], data['Highlights'] = processor.clean_data(data)
    print('Dataset Cleaned.')
    
    # remove long stories
    data['Stories'], data['Highlights'] = processor.remove_long_sequences(data)
    print('Long Stories Removed.')
    
    # remove duplicates and na
    data = processor.drop_dulp_and_na(data, ['Stories', 'Highlights'])
    print("Duplicates and NaN dropped.")
        
    # start and end tokens
    data['Highlights'] = processor.start_end_token(data['Highlights'])
    print("Start and End Tokens added.")
    
    # Tokenizer
    total_word, rare_word = processor.rare_words_count(data['Stories'])
    x_seq, x_tokenizer = processor.text2seq(data['Stories'], total_word, rare_word)
    x_seq = processor.pad_seq(x_seq, processor.max_length_story)
    
    total_word, rare_word = processor.rare_words_count(data['Highlights'])
    y_seq, y_tokenizer = processor.text2seq(data['Highlights'], total_word, rare_word)
    y_seq = processor.pad_seq(y_seq, processor.max_length_highlight)
    ("Tokenization Completed.")
    
    # Tokenizer Data
    x_vocab_size, y_vocab_size, input_word_index, target_word_index, reversed_input_word_index, reversed_target_word_index = processor.required_dicts(x_tokenizer,y_tokenizer)
    ("Tokenizer Data Loaded.")
    
    # split data
    x_tr, x_test, x_dev, y_tr, y_test, y_dev = processor.split_data(x_seq, y_seq, train_ratio=0.1, dev_ratio=0)
    print("Data Splitted.")
    
    # Pickle data required for building model
    processor.pickle_data([x_tr, x_test, x_dev, y_tr, y_test, y_dev], 'DataSequences')
    print("Data Sequences Pickled.")
    
    processor.pickle_data([x_tokenizer, y_tokenizer, x_vocab_size, y_vocab_size, input_word_index, target_word_index, 
    reversed_input_word_index, reversed_target_word_index,
    processor.max_length_story, processor.max_length_highlight], 'TokenizerData')
    print("Tokenizer Data Pickled.")
    
    """Model Building""" 
    from Summarizer import Summarizer
    summarizer = Summarizer()
    
    # Read in glove embeddinsg
    embeddings_index = summarizer.read_glove_embeddings()
    print("Embedding Vectors Loaded.")
    
    # embedding matrix
    embedding_matrix_input, embedding_matrix_target = summarizer.embedding_matrix(embeddings_index)
    print("Embedding Matrix Created.")
    
    # Define model
    trainer_model, encoder_model, decoder_model = summarizer.define_models(embedding_matrix_input,embedding_matrix_target)
    print("Model Defined.")
    
    # Compile model
    summarizer.compile_model(trainer_model)
    print("Model Compiled.")
    
    # Train model
    history = summarizer.train_model(trainer_model, x_tr, x_dev, y_tr, y_dev)
    print("Model Trained.")
    
    # Disgnostic plot
    print("Diagnostic Plot: ")
    summarizer.diagnostic_plot(history)
    
    # Save model
    summarizer.save_model(encoder_model, decoder_model)
    print("Model Saved.")
    
    """Predictions"""
    from Prediction import Prediction
    predictor = Prediction()
    
    # Load trained model
    encoder_model = predictor.load_model('encoder_model.json', 'encoder_model_weights.h5')
    decoder_model = predictor.load_model('decoder_model.json', 'decoder_model_weights.h5')
    print("Model Loaded.")
    
    # Generate summaries
    predictor.generated_summaries(3, encoder_model, decoder_model)


# In[ ]:





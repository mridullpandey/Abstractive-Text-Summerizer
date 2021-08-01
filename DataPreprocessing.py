try:
    import numpy as np
    import pandas as pd
    import string
    from sklearn.model_selection import train_test_split
    from nltk.corpus import stopwords
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    import nltk
    import pickle
    from TextCleaner import TextCleaner
except ImportError as e:
    print(e)


class DataPreprocessing():
    def __init__(self):
        self.max_length_text = 80
        self.max_length_summary = 10
    
    def load_dataset(self,filename,nrows):
        data = pd.read_csv(filename + '.csv',nrows = nrows)
        
        return data
    
    def remove_columns(self,data):
        data = data.drop(['Id','ProductId','UserId','ProfileName', 'HelpfulnessNumerator','HelpfulnessDenominator', 'Score', 'Time'],1)
        
        return data

    
    def clean_data(self,data):
        cleaner = TextCleaner()
        cleaned_text = []
        for t in data['Text']:
            cleaned_text.append(cleaner.text_preprocessing(str(t)))

        cleaned_summaries = []
        for s in data['Summary']:
            cleaned_summaries.append(cleaner.text_preprocessing(str(s), stop_words=False))
        
        data['Text'] = pd.Series(cleaned_text)
        data['Summary'] = pd.Series(cleaned_summaries)
        
        return data['Text'], data['Summary']
    
    def data_distribution(self,data):
        # Understanding distribution
        text_word_count = []
        for story in data['Text']:
            text_word_count.append(len(story.split()))

        summary_word_count = []
        for highlight in data['Summary']:
            summary_word_count.append(len(highlight.split()))
            
        return text_word_count, summary_word_count
    
    def max_length_graph(self,text_word_count, summary_word_count):
        length_df = pd.DataFrame({'text':text_word_count, 'summary':summary_word_count})
        length_df.hist(bins = 30)
        plt.show()
        
    def remove_long_sequences(self, data):
        text = np.array(data['Text'])
        summaries = np.array(data['Summary'])

        short_text = []
        short_summaries = []

        for i in range(len(summaries)):
            if (len(summaries[i].split()) <= self.max_length_summary) and (len(text[i].split()) <= self.max_length_text):
                short_text.append(text[i])
                short_summaries.append(summaries[i])
    
        data['Text'] = pd.Series(short_text)
        data['Summary'] = pd.Series(short_summaries)
    
        return data['Text'], data['Summary']
    
    def drop_dulp_and_na(self, data, columns, do_inplace=True):
        for subset in columns:
            data.drop_duplicates(subset=[subset], inplace=do_inplace)  # dropping duplicates
        data.replace('', np.nan, inplace=True)
        data.dropna(axis=0, inplace=do_inplace)
        
        return data
    
    def start_end_token(self, column):
        return column.apply(lambda x : 'sostok ' + str(x) + ' eostok') 
    
    def rare_words_count(self, data, thresh=5):
        data_tokenizer = Tokenizer()
        data_tokenizer.fit_on_texts(list(data))

        cnt = 0
        tot_cnt = 0
        freq = 0
        tot_freq = 0

        for key, value in data_tokenizer.word_counts.items():
            tot_cnt = tot_cnt + 1
            tot_freq = tot_freq + value
            if value < thresh:
                cnt = cnt + 1
                freq = freq + value

        return tot_cnt, cnt
    
    def text2seq(self, data, tot_cnt, cnt):
        # prepare a tokenizer for reviews on training data
        data_tokenizer = Tokenizer(num_words=tot_cnt - cnt)
        data_tokenizer.fit_on_texts(list(data))

        # convert text sequences into integer sequences
        data_seq = data_tokenizer.texts_to_sequences(data)


        return data_seq, data_tokenizer
    
    def pad_seq(self, data, maxlength, padding='post'):
        return pad_sequences(data, maxlen=maxlength, padding=padding)
    
    def required_dicts(self, x_tokenizer, y_tokenizer):
        # Vocabulary
        x_vocab_size = len(x_tokenizer.word_index) + 1
        y_vocab_size = len(y_tokenizer.word_index) + 1
        
        # Word to index
        input_word_index = x_tokenizer.word_index
        target_word_index = y_tokenizer.word_index
        
        # Index to word
        reversed_input_word_index = x_tokenizer.index_word
        reversed_target_word_index = y_tokenizer.index_word

        return x_vocab_size, y_vocab_size, input_word_index, target_word_index, reversed_input_word_index, reversed_target_word_index
    
    
    def split_data(self, X, y, train_ratio, dev_ratio, random=0, do_shuffle=True):
        X_tr, X_test, y_tr, y_test = train_test_split(np.array(X), np.array(y), test_size=(1 - train_ratio),
                                                      random_state=random, shuffle=do_shuffle)
        dev_len = int(dev_ratio * len(X))
        X_dev, X_test, y_dev, y_test = X_test[:dev_len], X_test[dev_len:], y_test[:dev_len], y_test[dev_len:]

        return X_tr, X_test, X_dev, y_tr, y_test, y_dev
    
    def pickle_data(self, data, filename, path=''):
        pickle_out = open(path + filename + '.pickle', "wb")
        pickle.dump(data, pickle_out)
        pickle_out.close()

    def load_pickle(self, filename, path=''):
        pickle_in = open(path + filename + '.pickle', "rb")
        return pickle.load(pickle_in)


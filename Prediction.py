from tensorflow.keras.models import model_from_json
from TextCleaner import TextCleaner
from attention import AttentionLayer
import numpy as np

class Prediction():
    def __init__(self, loaded_data):
        self.x_tokenizer = loaded_data[0]
        self.y_tokenizer = loaded_data[1]
        self.input_word_index = loaded_data[4]
        self.target_word_index = loaded_data[5]
        self.reversed_input_word_index = loaded_data[6]
        self.reversed_target_word_index = loaded_data[7]
        self.max_length_text = loaded_data[8]
        self.max_length_summary = loaded_data[9]

    def load_model(self, model_filename, model_weights_filename):
        with open(model_filename, 'r', encoding='utf8') as f:
            model = model_from_json(f.read(), custom_objects={'AttentionLayer': AttentionLayer})
        model.load_weights(model_weights_filename)
        return model

    def decode_sequence(self, input_seq, encoder_model, decoder_model):
        encoder_outputs, h0, c0 = encoder_model.predict(input_seq)

        # First word to be passed to the decoder 
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.target_word_index['sostok']

        decoded_sentence = ''
        stop_condition = False

        while not stop_condition:

            output_tokens, h, c = decoder_model.predict([target_seq] + [encoder_outputs, h0, c0])

            # Sample output tokens
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = self.reversed_target_word_index[sampled_token_index]

            if (sampled_word != 'eostok'):
                decoded_sentence += sampled_word + ' '

            if (sampled_word == 'eostok' or (len(decoded_sentence.split()) + 1) > self.max_length_summary):
                stop_condition = True

            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            h0, c0 = h, c

        return decoded_sentence

    def seqtotext(self, input_seq):
        text = ''
        for i in input_seq:
            if i != 0:
                text += self.reversed_input_word_index[i] + ' '

        return text

    def seqtosummary(self, input_seq):
        summary = ''
        for i in input_seq:
            if ((i != 0 and i != self.target_word_index['sostok']) and i != self.target_word_index['eostok']):
                summary = summary + self.reversed_target_word_index[i] + ' '
        return summary

    def text2seq(self, text):
        cleaner = TextCleaner()
        text = cleaner.text_preprocessing(text)
        seq = np.zeros(self.max_length_text)
        text = text.split()
        for x in range(len(text)):
            seq[x] = self.input_word_index[text[x]]

        return seq

    def generated_summaries(self, text, encoder_model, decoder_model):
        """Generates summary from given seq and models."""
        seq = self.text2seq(text)
        return self.decode_sequence(seq.reshape(1, self.max_length_text), encoder_model, decoder_model)
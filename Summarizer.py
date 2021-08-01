from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from tensorflow.keras.models import Model
from attention import AttentionLayer
from DataPreprocessing import DataPreprocessing
from TextCleaner import TextCleaner
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
class Summarizer():
    def __init__(self,input_word_index, target_word_index, x_vocab_size, y_vocab_size):
        self.input_word_index = input_word_index
        self.target_word_index = target_word_index
        self.x_vocab_size = x_vocab_size
        self.y_vocab_size = y_vocab_size
        self.latent_dim = 128
        self.embedding_dim = 100

    def read_glove_embeddings(self):
        embeddings_index = dict()
        f = open('glove.6B.100d.txt', encoding='utf8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        return embeddings_index

    def embedding_matrix(self, embeddings_index):
        embedding_matrix_input = np.zeros((self.x_vocab_size, 100))

        for word, idx in self.input_word_index.items():
            embedding_vector = embeddings_index.get(word)

            if embedding_vector is not None:
                embedding_matrix_input[idx, :] = embedding_vector
            else:
                new_embedding = np.random.uniform(-1, 1, (1, 100))
                embedding_matrix_input[idx, :] = new_embedding

        embedding_matrix_target = np.zeros((self.y_vocab_size, 100))

        for word, idx in self.target_word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix_target[idx, :] = embedding_vector
            else:
                new_embedding = np.random.uniform(-1, 1, (1, 100))
                embedding_matrix_target[idx, :] = new_embedding

        return embedding_matrix_input, embedding_matrix_target

    def define_models(self, embedding_matrix_input, embedding_matrix_target, max_length_text, max_length_summary):
        """Training Phase"""
        # Encoder
        encoder_inputs = Input(shape=(max_length_text,))
        enc_emb = Embedding(self.x_vocab_size, self.embedding_dim, weights=[embedding_matrix_input],
                            input_length=max_length_text, trainable=False)(encoder_inputs)
        encoder = Bidirectional(
            LSTM(self.latent_dim, return_sequences=True, return_state=True, dropout=0.3, recurrent_dropout=0.3))
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(enc_emb)
        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])

        # Decoder
        decoder_inputs = Input(shape=(None,))
        dec_emb_layer = Embedding(self.y_vocab_size, self.embedding_dim, weights=[embedding_matrix_target],
                                  input_length=max_length_summary, trainable=False)
        dec_emb = dec_emb_layer(decoder_inputs)
        decoder_lstm = LSTM(2 * self.latent_dim, return_sequences=True, return_state=True, dropout=0.3,
                            recurrent_dropout=0.3)
        decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

        # Attention
        attn_layer = AttentionLayer(name='attention_layer')
        attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

        # Concatenate the context vectors with the decoder outpouts
        decoder_concat = Concatenate()([decoder_outputs, attn_out])

        # Dense
        decoder_dense = TimeDistributed(Dense(self.y_vocab_size, activation='softmax'))
        decoder_outputs = decoder_dense(decoder_concat)

        # model
        trainer_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)

        """Inference Phase"""
        # Encoder
        encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])

        # Decoder
        decoder_state_input_h = Input(shape=(2 * self.latent_dim,))
        decoder_state_input_c = Input(shape=(2 * self.latent_dim,))
        decoder_hidden_state_input = Input(shape=(max_length_text, 2 * self.latent_dim))
        dec_emb2 = dec_emb_layer(decoder_inputs)
        decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h,
                                                                                     decoder_state_input_c])

        # Attention 
        attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
        decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

        # Dense
        decoder_outputs2 = decoder_dense(decoder_inf_concat)
        decoder_model = Model(
            [decoder_inputs] + [decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c],
            [decoder_outputs2] + [state_h2, state_c2])

        return trainer_model, encoder_model, decoder_model

    def compile_model(self, model, optimizer='adam', loss='sparse_categorical_crossentropy'):
        model.compile(optimizer, loss)

    def train_model(self, model, x_tr, x_dev, y_tr, y_dev, epochs=50, batch_size=128):
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
        history = model.fit([x_tr, y_tr[:, :-1]], y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)[:, 1:], epochs=epochs,
                            callbacks=[es], batch_size=batch_size, validation_data=(
            [x_dev, y_dev[:, :-1]], y_dev.reshape(y_dev.shape[0], y_dev.shape[1], 1)[:, 1:]))

        return history

    def diagnostic_plot(self, history):
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='dev')
        plt.legend()
        plt.show()

    def save_model(self, encoder_model, decoder_model):
        with open('encoder_model.json', 'w', encoding='utf8') as f:
            f.write(encoder_model.to_json())
        encoder_model.save_weights('encoder_model_weights.h5')

        with open('decoder_model.json', 'w', encoding='utf8') as f:
            f.write(decoder_model.to_json())
        decoder_model.save_weights('decoder_model_weights.h5')

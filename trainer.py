import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import librosa
import time
from tqdm import tqdm
import edit_distance as ed


from model.configs import SR, device_name, UNQ_CHARS, INPUT_DIM, MODEL_NAME, NUM_UNQ_CHARS
from model.utils import CER_from_mfccs, batchify, clean_single_wav, gen_mfcc, indices_from_texts, load_model
from model.model import get_model


def train_model(model, optimizer, train_wavs, train_texts, test_wavs, test_texts, epochs=100, batch_size=50):

    with tf.device(device_name):

        for e in range(0, epochs):
            start_time = time.time()

            len_train = len(train_wavs)
            len_test = len(test_wavs)
            train_loss = 0
            test_loss = 0
            test_CER = 0
            train_batch_count = 0
            test_batch_count = 0

            print("Training epoch: {}".format(e+1))
            for start in tqdm(range(0, len_train, batch_size)):

                end = None
                if start + batch_size < len_train:
                    end = start + batch_size
                else:
                    end = len_train
                x, target, target_lengths, output_lengths = batchify(
                    train_wavs[start:end], train_texts[start:end], UNQ_CHARS)

                with tf.GradientTape() as tape:
                    output = model(x, training=True)

                    loss = K.ctc_batch_cost(
                        target, output, output_lengths, target_lengths)

                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                train_loss += np.average(loss.numpy())
                train_batch_count += 1

            print("Testing epoch: {}".format(e+1))
            for start in tqdm(range(0, len_test, batch_size)):

                end = None
                if start + batch_size < len_test:
                    end = start + batch_size
                else:
                    end = len_test
                x, target, target_lengths, output_lengths = batchify(
                    test_wavs[start:end], test_texts[start:end], UNQ_CHARS)

                output = model(x, training=False)

                # Calculate CTC Loss
                loss = K.ctc_batch_cost(
                    target, output, output_lengths, target_lengths)

                test_loss += np.average(loss.numpy())
                test_batch_count += 1

                """
                    The line of codes below is for computing evaluation metric (CER) on internal validation data.
                """
                input_len = np.ones(output.shape[0]) * output.shape[1]
                decoded_indices = K.ctc_decode(output, input_length=input_len,
                                       greedy=False, beam_width=100)[0][0]
                
                # Remove the padding token from batchified target texts
                target_indices = [sent[sent != 0].tolist() for sent in target]

                # Remove the padding, unknown token, and blank token from predicted texts
                predicted_indices = [sent[sent > 1].numpy().tolist() for sent in decoded_indices] # idx 0: padding token, idx 1: unknown, idx -1: blank token

                len_batch = end - start
                for i in range(len_batch):

                    pred = predicted_indices[i]
                    truth = target_indices[i]
                    sm = ed.SequenceMatcher(pred, truth)
                    ed_dist = sm.distance()                 # Edit distance
                    test_CER += ed_dist / len(truth)
                test_CER /= len_batch

            train_loss /= train_batch_count
            test_loss /= test_batch_count
            test_CER /= test_batch_count

            rec = "Epoch: {}, Train Loss: {:.2f}, Test Loss {:.2f}, Test CER {:.2f} % in {:.2f} secs.\n".format(
                e+1, train_loss, test_loss, test_CER*100, time.time() - start_time)

            print(rec)


def load_data(wavs_dir, texts_dir):
    texts_df = pd.read_csv(texts_dir)
    train_wavs = []
    for f_name in texts_df["file"]:
        wav, _ = librosa.load(f"{wavs_dir}/{f_name}.flac", sr=SR)
        train_wavs.append(wav)
    train_texts = texts_df["text"].tolist()
    return train_wavs, train_texts


if __name__ == "__main__":

    # Defintion of the model
    model = get_model(INPUT_DIM, NUM_UNQ_CHARS, num_res_blocks=5, num_cnn_layers=2,
                      cnn_filters=50, cnn_kernel_size=15, rnn_dim=170, rnn_dropout=0.15, num_rnn_layers=2,
                      num_dense_layers=1, dense_dim=340, model_name=MODEL_NAME, rnn_type="lstm",
                      use_birnn=True)
    print("Model defined \u2705 \u2705 \u2705 \u2705\n")

    # Defintion of the optimizer
    optimizer = tf.keras.optimizers.Adam()

    # Load the data
    print("Loading data.....")
    train_wavs, train_texts = load_data(
        wavs_dir="dataset/wav_files(sampled)", texts_dir="dataset/transcriptions(sampled)/file_speaker_text(sampled).csv")
    print("Data loaded \u2705 \u2705 \u2705 \u2705\n")

    """
    To replicate the results give the argument as text_dir="dataset/transcriptions(sampled)/file_speaker_text(orignally_trained).csv".
    Get all of the wavs files from https://openslr.org/54/, put them in a single directory, and give that directory as argument for wavs_dir.
    """
    
    # Clean the audio file by removing the silent gaps from the both ends the audio file
    print("Cleaning the audio files.....")
    train_wavs = [clean_single_wav(wav) for wav in train_wavs]
    print("Audio files cleaned \u2705 \u2705 \u2705 \u2705\n")

    # Generate mfcc features for the audio files
    print("Generating mfcc features.....")
    train_wavs = [gen_mfcc(wav) for wav in train_wavs]
    print("MFCC features generated \u2705 \u2705 \u2705 \u2705\n")

    # Train Test Split
    """
    Originally the data was split in the 95% train and 5% test set; With total of 148K (audio,text) pairs.
    """
    train_wavs, test_wavs, train_texts, test_texts = train_test_split(
        train_wavs, train_texts, test_size=0.2)

    # Train the model
    """
    Originally the model was trained for 58 epochs; With a batch size of 50.
    """
    train_model(model, optimizer, train_wavs, train_texts,
                test_wavs, test_texts, epochs=100, batch_size=2)

# U.0
# Loads model from the directory argument
# def load_model(model_dir):

# U.1
# Loading wav file from librosa
# def load_wav(dir):

# U.2
# Generates Normalized MFCCs from audio
# def gen_mfcc(arr):

# U.3
# Generates padded text from list of texts
# def pad_text(list_texts, unq_chars, unk_idx=1):

# U.4
# Returns tensor batch*seq*frame
# def pad_list_np(list_np):

# U.5
# Generates batches of wavs and texts  with padding as per needed
# def batchify(wavs, texts, unq_chars):

# U.6
# Plots lossses from the file
# def plot_losses(dir, optimal_epoch=None):


# U.7
# Decoding with prefix beam search from MFCC features only
# def predict_from_mfccs(model, mfccs, unq_chars):

# U.8
# Decoding with prefix beam search from wavs only
# def predict_from_wavs(model, wavs, unq_chars):

# U.9
# Converts the text to list of indices as per the unique characters list
# def indices_from_texts(texts_list, unq_chars, unk_idx=1):

# U.10
# CER from mfccs
# def CER_from_mfccs(model, mfccs, texts, unq_chars, batch_size=100):

# U.11
# CER from wavs
# def CER_from_wavs(model, wavs, texts, unq_chars, batch_size=100):

# U.12
# CTC softmax probabilities output from mfcc features
# def ctc_prob_output_from_mfccs(model, mfccs):

# U.13
# CTC softmax probabilities output from wavs
# def ctc_prob_output_from_wavs(model, wavs):

# U.14
# Clean the single audio file by clipping silent gaps from both ends
# def clean_single_wav(wav, win_size=500):

'''
Important Notes:
    1. The dimension of wav files is in the format of  -> batch * sequence * frame_size.

'''


from tensorflow.keras import models
import tensorflow.keras.backend as K
import tensorflow as tf
import librosa
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time
import edit_distance as ed


# Global configs required for training
from .configs import INPUT_DIM, SR, N_MFCC, HOP_LENGTH, FRAME_SIZE

device_name = '/device:CPU:0'

# U.0
# Loads model from the directory argument


def load_model(model_dir):
    return models.load_model(model_dir)

# U.1
# Loading wav file from librosa


def load_wav(dir):
    return librosa.load(dir, sr=SR)[0]

# U.2
# Generates Normalized MFCCs from audio


def gen_mfcc(arr):
    mfccs = librosa.feature.mfcc(
        y=arr[:-1], sr=SR, n_mfcc=N_MFCC, hop_length=HOP_LENGTH).transpose().flatten()
    return (mfccs - np.mean(mfccs)) / np.std(mfccs)


# U.3
# Generates padded text from list of texts
def pad_text(list_texts, unq_chars, unk_idx=1):
    max_len = max([len(txt) for txt in list_texts])
    padded_arr = []
    seq_lengths = []

    for txt in list_texts:
        len_seq = len(txt)
        txt += "0" * (max_len - len_seq)

        # index 1 for the unknown chars
        arr = np.array(
            [unq_chars.index(ch) if ch in unq_chars else unk_idx for ch in txt])

        padded_arr.append(arr)
        seq_lengths.append(len_seq)

    return np.array(padded_arr), np.array(seq_lengths)

# U.4
# Returns tensor batch*seq*frame


def pad_list_np(list_np):
    max_len = max([len(arr) for arr in list_np])

    # So that the numpy array can be reshaped according to the input dimension
    max_len += INPUT_DIM - (max_len % INPUT_DIM)

    padded_arr = []

    for arr in list_np:
        len_seq = len(arr)
        arr = np.pad(arr, (0, max_len - len_seq), constant_values=0)
        padded_arr.append(arr)

    return np.array(padded_arr).reshape((len(list_np), -1, INPUT_DIM))

# U.5
# Generates batches of wavs and texts  with padding as per needed


def batchify(wavs, texts, unq_chars):
    assert len(wavs) == len(texts)
    # generates tensor of dim (batch * seq * frame)
    input_tensor = pad_list_np(wavs)
    target_tensor, target_lengths_tensor = pad_text(texts, unq_chars)
    output_seq_lengths_tensor = np.full(
        (len(wavs),), fill_value=input_tensor.shape[1])

    return input_tensor, target_tensor, target_lengths_tensor.reshape((-1, 1)), output_seq_lengths_tensor.reshape((-1, 1))


# U.6
# Plots lossses from the file
def plot_losses(dir, optimal_epoch=None):
    losses = None
    with open(dir, "rb") as f:
        losses = pickle.load(f)
        f.close()

    train_losses, test_losses = losses["train_losses"], losses["test_losses"]
    epochs = len(train_losses)
    # print(len(test_losses))

    X = range(1, epochs+1)

    fig, ax = plt.subplots(1, figsize=(15, 10))

    fig.suptitle('Train and Test Losses', fontsize=25,)

    ax.set_xlim(0, 72)
    ax.plot(X, train_losses, color="red", label="Train Loss")
    ax.plot(X, test_losses, color="green", label="Test Loss")

    plt.rcParams.update({'font.size': 20})

    plt.legend(loc="upper right", frameon=False, fontsize=20)
    # plt.xlabel("Epochs",{"size":20})
    # plt.ylabel("Loss", {"size":20})

    if (optimal_epoch != None):
        plt.axvline(x=optimal_epoch, ymax=0.5)
        # ax.plot(58, 0, 'go', label='marker only')
        plt.text(optimal_epoch, 35,
                 f'Optimal Epoch at {optimal_epoch}', fontsize=15)

    plt.show()


# U.7
# Decoding with prefix beam search from MFCC features only
def predict_from_mfccs(model, mfccs, unq_chars):

    mfccs = pad_list_np(mfccs)  # coverts the data to 3d
    pred = model(mfccs, training=False)

    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = K.ctc_decode(pred, input_length=input_len,
                           greedy=False, beam_width=100)[0][0]

    sentences = []
    char_indices = []
    for chars_indices in results:

        sent = ""
        temp_indices = []
        for idx in chars_indices:

            if idx > 1:
                sent += unq_chars[idx]
                temp_indices.append(idx.numpy())
        sentences.append(sent)
        char_indices.append(temp_indices)
    return sentences, char_indices


# U.8
# Decoding with prefix beam search from wavs only
def predict_from_wavs(model, wavs, unq_chars):
    mfccs = [gen_mfcc(wav) for wav in wavs]
    return predict_from_mfccs(model, mfccs, unq_chars)

# U.9
# Converts the text to list of indices as per the unique characters list


def indices_from_texts(texts_list, unq_chars, unk_idx=1):

    indices_list = []
    for txt in texts_list:

        # index 1 for the unknown chars
        lst = [unq_chars.index(
            ch) if ch in unq_chars else unk_idx for ch in txt]

        indices_list.append(lst)

    return indices_list


'''
Calculates CER( character error rate) from dataset;
'''
# U.10
# CER from mfccs


def CER_from_mfccs(model, mfccs, texts, unq_chars, batch_size=100):

    with tf.device(device_name):

        len_mfccs = len(mfccs)
        batch_count = 0
        sum_cer = 0

        start_time = time.time()
        for start in range(0, len_mfccs, batch_size):
            end = None
            if start + batch_size < len_mfccs:
                end = start + batch_size
            else:
                end = len_mfccs
            pred_sentences, pred_indices = predict_from_mfccs(
                model, mfccs[start:end], unq_chars)
            actual_indices = indices_from_texts(texts[start:end], unq_chars)

            len_batch_texts = end - start
            batch_cer = 0
            for i in range(len_batch_texts):

                pred = pred_indices[i]
                actu = actual_indices[i]

                sm = ed.SequenceMatcher(pred, actu)
                ed_dist = sm.distance()
                batch_cer += ed_dist / len(actu)

            batch_cer /= len_batch_texts
            batch_count += 1
            sum_cer += batch_cer

            print("CER -> {:.2f}%, \t No.of sentences -> {}, \t Time Taken -> {:.2f} secs.".format(
                (sum_cer / batch_count) * 100, end, time.time() - start_time))

        print(
            "The total time taken for all sentences CER calculation is  {:.2f} secs.".format(time.time() - start_time))
        return sum_cer / batch_count

# U.11
# CER from wavs


def CER_from_wavs(model, wavs, texts, unq_chars, batch_size=100):

    assert len(wavs) == len(texts)

    len_wavs = len(wavs)
    for i in range(len_wavs):
        wavs[i] = gen_mfcc(wavs[i])

    return CER_from_mfccs(model, wavs, texts, unq_chars, batch_size)


# U.12
# CTC softmax probabilities output from mfcc features
def ctc_softmax_output_from_mfccs(model, mfccs):
    mfccs = pad_list_np(mfccs)
    y = model(mfccs)
    return y

# U.13
# CTC softmax probabilities output from wavs


def ctc_softmax_output_from_wavs(model, wavs):
    mfccs = [gen_mfcc(wav) for wav in wavs]
    return ctc_softmax_output_from_mfccs(model, mfccs)


# U.14
# Clean the single audio file by clipping silent gaps from both ends
def clean_single_wav(wav, win_size=500):
    wav_avg = np.average(np.absolute(wav))

    for s in range(0, len(wav)-win_size, win_size):
        window = wav[s:s+win_size]
        if np.average(np.absolute(window)) > wav_avg:
            wav = wav[s:]
            break

    for s in range(len(wav)-win_size, 0, -win_size):
        window = wav[s-win_size:s]
        if np.average(np.absolute(window)) > wav_avg:
            wav = wav[:s]
            break

    pad = FRAME_SIZE - len(wav) % FRAME_SIZE
    wav = np.pad(wav, (0, pad), mode="mean")
    return wav

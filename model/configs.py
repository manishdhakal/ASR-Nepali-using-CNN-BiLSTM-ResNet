import tensorflow as tf


# Necessary global configs
LOAD_MFCC_FILES = True

FRAME_SIZE = 160
SR = 16000
FRAME_RATE = int( SR / FRAME_SIZE )
N_MFCC = 13
HOP_LENGTH = 40

assert FRAME_SIZE % HOP_LENGTH == 0

INPUT_DIM = int(N_MFCC * (FRAME_SIZE / HOP_LENGTH))

UNQ_CHARS = [' ', 'ँ', 'ं', 'ः', 'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ए', 'ऐ', 'ओ', 'औ', 'क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श', 'ष', 'स', 'ह', 'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'े', 'ै', 'ो', 'ौ', '्', 'ॠ', '\u200c', '\u200d', '।']
UNQ_CHARS = ['0', 'u' ] + sorted(UNQ_CHARS) + ['-'] #"0" -> padding char,"u" -> unknown chars "-" -> blank char
NUM_UNQ_CHARS = len(UNQ_CHARS) # +1 is for '-' blank at last

MODEL_NAME = "ASR_model"

# ALL of the directories for weights, model and losses are handled here
# import os

# DATASET_DIR = "/content/drive/My Drive/datasets/ASR-NP/cleaned_datasets"

# if LOAD_MFCC_FILES :
#     DATASET_DIR = "/content/drive/My Drive/datasets/ASR-NP/13_mel_scales_mfcc_cleaned_datasets"

# MODEL_NAME = "CNN + ResNet + BiLSTM"
# MAIN_DIR = f"/content/drive/My Drive/models/ASR_NP/{MODEL_NAME}"
# HISTORY_DIR = f"{MAIN_DIR}/history.txt"
# WEIGHTS_DIR = f"{MAIN_DIR}/weights"
# SAVED_MODEL_DIR = f"{MAIN_DIR}/saved_model/trained_model"
# LOSSES_DIR = f"{MAIN_DIR}/losses.pickle"

# if not os.path.exists(MAIN_DIR):
#     os.makedirs(MAIN_DIR)
# if not os.path.exists(WEIGHTS_DIR):
#     os.makedirs(WEIGHTS_DIR)
# if not os.path.exists(f"{MAIN_DIR}/saved_model"):
#     os.makedirs(f"{MAIN_DIR}/saved_model")



# Checks for the availability of the GPU
device_name = tf.test.gpu_device_name()

if device_name != '/device:GPU:0':
    device_name = '/device:CPU:0'
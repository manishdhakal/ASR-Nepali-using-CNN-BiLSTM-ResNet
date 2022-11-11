"""
Author: Manish Dhakal
Year: 2022
Email: mns.dkl19@gmail.com
Publication: https://doi.org/10.1109/ICICT54344.2022.9850832
"""

from .configs import MODEL_NAME, INPUT_DIM, NUM_UNQ_CHARS
from tensorflow.keras import layers, Model, Input
import tensorflow.keras.backend as K
import numpy as np


# Residual block for the model
def res_block(ip, num_cnn_layers, cnn_filters, cnn_kernel_size, use_resnet):

    res_arch = ip
    for _ in range(num_cnn_layers):
        res_arch = layers.Conv1D(
            cnn_filters, cnn_kernel_size, padding='same')(res_arch)
        res_arch = layers.BatchNormalization()(res_arch)
        res_arch = layers.PReLU(shared_axes=[1])(res_arch)

    return layers.add([ip, res_arch]) if use_resnet else res_arch


# Main ASR model
def get_model(ip_channel, num_classes, num_res_blocks=3, num_cnn_layers=1, cnn_filters=50,
             cnn_kernel_size=15,  num_rnn_layers=2, rnn_dim=170, num_dense_layers=1,
             dense_dim=300, use_birnn=True, use_resnet=True, rnn_type="lstm", rnn_dropout=0.15,
             model_name=None):

    input = Input(shape=(None, ip_channel))

    arch = layers.Conv1D(cnn_filters, cnn_kernel_size, padding="same")(input)
    arch = layers.BatchNormalization()(arch)
    arch = layers.PReLU(shared_axes=[1])(arch)

    for _ in range(num_res_blocks):
        arch = res_block(arch, num_cnn_layers, cnn_filters,
                         cnn_kernel_size, use_resnet)

    rnn = layers.GRU if rnn_type == "gru" else layers.LSTM

    for _ in range(num_rnn_layers):
        if use_birnn:
            arch = layers.Bidirectional(
                rnn(rnn_dim, dropout=rnn_dropout, return_sequences=True))(arch)
        else:
            arch = rnn(rnn_dim, dropout=rnn_dropout,
                       return_sequences=True)(arch)

    for _ in range(num_dense_layers):
        arch = layers.Dense(dense_dim)(arch)
        arch = layers.ReLU()(arch)

    arch = layers.Dense(num_classes)(arch)
    output = K.softmax(arch)

    model = Model(inputs=input, outputs=output, name=model_name)

    return model


if __name__ == "__main__":
    # Defintion of the model
    model = get_model(INPUT_DIM, NUM_UNQ_CHARS, num_res_blocks=5, num_cnn_layers=2,
                     cnn_filters=50, cnn_kernel_size=15, rnn_dim=170, num_rnn_layers=2,
                     num_dense_layers=1, dense_dim=340, model_name=MODEL_NAME, rnn_type="lstm",
                     use_birnn=True)
    x = np.random.rand(2, 100, INPUT_DIM)
    y = model(x)
    print(y.shape)
    # model.summary()

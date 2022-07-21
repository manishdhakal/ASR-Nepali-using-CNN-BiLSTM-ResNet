# Automatic Speech Recognition for the Nepali Language using CNN, Bidirectional LSTM and, ResNet
## Keywords
```Nepali Speech Recognition, CNN, ResNet, BiLSTM, CTC ```
## Intorduction
This repo is a part of the [research](#) project for designing the automatic speech recogntion(ASR) model for Nepali language using ML techniques.

## Things to consider
- You are free to use this research as refernce and make modifications to continue your own research in Nepali ASR. 
- The `trainer.py` has been implemented to run on the sampled data for now. To replicate the result please replace the sampled dataset with original [OpenSLR dataset.](https://openslr.org/54)
- Please remove the (audio, text) pairs that include Devnagari numeric transcriptions like १४२३, ५९२, etc from the dataset because they degrade the performance of dataset.

## Our approach
0. Remove the (audio, text) pairs that include Devnagari numeric transcriptions
1. Data cleaning (clipping silent gaps from both ends)
2. MFCC feature extraction from audio data
3. Design Neural Network (optimal: CNN + ResNet + BiLSTM) model 
4. Calculate CTC loss for applying gradient (training)
5. Decode the texts by using beam search decoding (infernce)

## Architecture
![Res_block](https://github.com/manishdhakal/ASR-Nepali-using-CNN-BiLSTM-ResNet/blob/main/media/res_block.png?raw=true)

![Model](https://github.com/manishdhakal/ASR-Nepali-using-CNN-BiLSTM-ResNet/blob/main/media/model.png?raw=true)

## Running the project
0. Initialize the virtual environment by installing packages from `requirements.txt`.
1. Running training pipeline & `eval.py` can be also be used to evaluate your own (audio,text) pairs.
```
python trainer.py   # For running the training pipeline
python eval.py      # For testing and evaluating the model already trained by the author
```

## Results
Models and Their character error rate (CER) on Test Data (5% of Total Data.)

| Model | Test Data CER | # Params |
| :---: | :---: | :---: | 
|BiLSTM | 19.71% | 1.17M |
|  1D-CNN + BiLSTM | 24.6% | 1.55M |            
|  1D-CNN + ResNet + BiGRU | 29.6% | 1.30M |            
|  **1D-CNN + ResNet + BiLSTM** | **17.06%** | **1.55M**|
|  1D-CNN + ResNet + LSTM | 30.27% | 0.88M|

For any queries email the author [here](mailto:mns.dkl19@gmail.com).
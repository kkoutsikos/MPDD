# MM 2025 MPDD Baseline Code
The baseline system provided for the MM 2025 MPDD Challenge serves as a starting point for participants to develop their solutions for the Multimodal Personalized Depression Detection tasks. The baseline system is designed to be straightforward yet effective, providing participants with a solid foundation upon which they can build and improve.
# Results

The metrics reported are accuracy (Acc.) and F1-score, both with and without personalized features (PF) for the MPDD-Young and MPDD-Elderly datasets. Each value represents the best-performing feature combination for each experiment, using default hyper-parameters.

#### MPDD-Elderly(Track1)

| Length | Task Type | Audio Feature | Visual Feature | w/o PF (Acc./F1) | w/ PF (Acc./F1) |
|--------|-----------|---------------|----------------|-------------------|-----------------|
| 1s     | Binary    | mfcc          | openface       | 83.33 / 70.89     | 84.62 / 79.13   |
| 1s     | Ternary   | opensmile     | resnet         | 55.13 / 49.14     | 56.41 / 55.64   |
| 1s     | Quinary   | opensmile     | densenet       | 66.67 / 44.00     | 69.23 / 46.66   |
| 5s     | Binary    | opensmile     | resnet         | 76.92 / 66.15     | 80.77 / 72.37   |
| 5s     | Ternary   | wav2vec       | openface       | 50.00 / 47.59     | 57.69 / 59.37   |
| 5s     | Quinary   | mfcc          | densenet       | 75.64 / 56.83     | 78.21 / 58.40   |


#### MPDD-Young(Track2)

| Length | Task Type | Audio Feature | Visual Feature | w/o PF (Acc./F1) | w/ PF (Acc./F1) |
|--------|-----------|---------------|----------------|------------------|-----------------|
| 1s     | Binary    | wav2vec       | openface       | 56.06 / 55.23    | 63.64 / 59.96   |
| 1s     | Ternary   | mfcc          | densenet       | 48.48 / 43.72    | 51.52 / 51.62   |
| 5s     | Binary    | opensmile     | resnet         | 60.61 / 60.02    | 62.12 / 62.11   |
| 5s     | Ternary   | mfcc          | densenet       | 42.42 / 39.38    | 50.00 / 41.31   |

# Environment

    python >= 3.8.0
    pytorch >= 1.0.0
    scikit-learn = 1.5.1

# Feature Extraction

In our baseline, we use the following features:

### Acoustic Feature:
**Wav2vec：** We extract utterance-level acoustic features using the wav2vec model pre-trained on large-scale audio data. The embedding size of the acoustic features is 512.  
The link of the pre-trained model is: [wav2vec model](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec)

**MFCCs：** We extract Mel-frequency cepstral coefficients (MFCCs). The embedding size of MFCCs is 64.  

**OpenSmile：** We extract utterance-level acoustic features using opensmile. The embedding size of OpenSMILE features is 6373.  

### Visual Feature:
**Resnet-50 and Densenet-121：** We employ OpenCV tool to extract scene pictures from each video, capturing frames at a 10-frame interval. Subsequently, we utilize the Resnet-50 and Densenet-121 model to generate utterance-level features for the extracted scene pictures in the videos. The embedding size of the visual features is 1000.
The links of the pre-trained models are:  
- [ResNet-50](https://huggingface.co/microsoft/resnet-50)  
- [DenseNet-121](https://huggingface.co/pytorch/vision/v0.10.0/densenet121)  

**OpenFace：** We extract csv visual features using the pretrained OpenFace model. The embedding size of OpenFace features is 709. You can download the executable file and model files for OpenFace from the following link: [OpenFace Toolkit](https://github.com/TadasBaltrusaitis/OpenFace)

### Personalized Feature:
We generate personalized features by loading the GLM3 model, creating personalized descriptions, and embedding these descriptions using the `roberta-large` model. The embedding size of the personalized features is 1024.  
The link of the `roberta-large` model is: [RoBERTa Large](https://huggingface.co/roberta-large)

# Usage
## Feature Extraction

First, you should update the data paths in the feature extraction files within the corresponding feature extraction folder. Then download the appropriate model files for wav2vec and OpenFace. Manually enter the following command to configure the environment:
   ```bash
  pip install scikit-image fire opencv-python tqdm matplotlib pandas soundfile wenetruntime fairseq==0.9.0 numpy==1.26.4 transformers paddlespeech pytest-runner paddlepaddle whisper -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```
For personalized features,
 install PyTorch (CUDA version) and Transformers:  
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
   ```bash
   pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers==4.40.0
   ```

- **gen_description.py**: This script loads the GLM3 model, reads the input data, and generates personalized descriptions.  
  - Modify the `generate_patient_prompt()` function to specify the required indicators for your dataset.  
  - Ensure the label file format matches the provided `label_data.json`.  

- **extrapersonality.py**: This script embeds the personalized descriptions using the `roberta-large` model to generate feature embeddings.

The obtained features and lables should be organized as follows, where `data_rootpath` is put into the script during training and testing.

`data_rootpath`:

    ├── Annotation/
    │   ├── Training_Validation_files.json
    │   ├── Testing_files.json
    ├── feature_personalized/
    │   ├── descriptions_embeddings_with_ids.npy
    ├── features/
    │   ├── densenet_1s/
    │   ├── densenet_5s/
    │   ├── mfccs_1s/
    │   ├── mfccs_5s/
    │   ├── openface_1s/
    │   ├── openface_5s/
    │   ├── opensmile_1s/
    │   ├── opensmile_5s/
    │   ├── ResNet_1s/
    │   ├── ResNet_5s/
    │   ├── wav2vec_1s/
    │   ├── wav2vec_5s/

## Training
To train the model with default parameters, taking MPDD-Young for example, simply run:  
```bash
bash scripts/Track2/train_1s_binary.sh
```

You can also modify parameters such as feature types, split window time, classification dimensions, or learning rate directly through the command line:  
```bash
bash scripts/Track2/train_1s_binary.sh --audiofeature_method=wav2vec --videofeature_method=resnet --splitwindow_time=5s --labelcount=5 --batch_size=32 --lr=0.001 --num_epochs=500
```
Refer to `config.json` for more parameters.

## Testing
To predict the lables for testing set with your obtained model, first modify the default parameters in `test.sh` to match the current task, and run:  
```bash
bash scripts/test.sh
```
After testing 6 tasks in Track1 or 4 tasks in Track2, the results will be merged into the `test.csv` file in `./answer_Track2/`.
Notice that the given lables in `Testing_Files.json` are not yet true lables, the actual test lables will be used for model evaluation at the later stage of the challenge.

# Acknowledgements
MPDD is developed based on the work of MEIJU 2025. The Github URL of MEIJU 2025 is: https://github.com/AI-S2-Lab/MEIJU2025-baseline.

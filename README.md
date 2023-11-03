# Proxemics-Net
Human interaction recognition in still images

Code prepared for paper presented at the IbPRIA'23 Conference.

## 1. File Structure

- `base_model_main/`: Main directory for the base model.
- `train/`: Directory containing code and resources related to model training.
- `test/`: Directory containing code and resources related to model testing.
- `dataset/`: Directory containing the code necessary for dataset preprocessing.
- `dataset_proxemics_IbPRIA.zip`: ZIP file containing the preprocessed dataset.
- `requirements.txt`: File specifying the necessary dependencies for the project.

  
## 2. Installing Dependencies

To install the necessary dependencies to run this project, you can use the following command:

    conda create --name <env> --file requirements.txt

## 3. Unzipping the Preprocessed Dataset ZIP

To use the preprocessed dataset, you must first unzip the `dataset_proxemics_IbPRIA.zip` file and place it two directories above the current directory. You can use the following command:

    unzip dataset_proxemics_IbPRIA.zip -d ../

## 4. Downloading Pre-Trained Models

To use pre-trained ConvNeXt models, you need to download them from the following locations:

- Pre-trained Base model: [Download here](https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth)
- Pre-trained Large model: [Download here](https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth)

Once downloaded, you need to unzip them and place them one level above, i.e., in ../premodels/.

## 5. Training a New Model

To train and test a new model, you should access the `base_model_main` directory and execute the following command lines based on the type of model you want to use:

### For Backbone Vit

#### Full Model (3 Branches)

    python3 base_model_main_ViT.py --datasetDIR <DIR dataset/> --outModelsDIR <DIR where you'll save the model> --b <batchsize> --set <set1/set2> --lr <learningRate>

#### Only Pair RGB

    python3 base_model_main_ViT.py --datasetDIR <DIR dataset/> --outModelsDIR <DIR where you'll save the model> --b <batchsize> --set <set1/set2> --lr <learningRate> --onlyPairRGB

### For Backbone ConvNeXt (Base or Large)

#### Full Model (3 Branches)

    python3 base_model_main_convNext.py --datasetDIR <DIR dataset/> --outModelsDIR <DIR where you'll save the model> --modeltype <base/large> --b <batchsize> --set <set1/set2> --lr <learningRate>

#### Only Pair RGB

    python3 base_model_main_convNext.py --datasetDIR <DIR dataset/> --outModelsDIR <DIR where you'll save the model> --modeltype <base/large> --b <batchsize> --set <set1/set2> --lr <learningRate> --onlyPairRGB

Be sure to adjust the values between <...> with the specific paths and configurations required for your project.

# Proxemics-Net

Code prepared for paper presented at the IbPRIA'23 Conference: [Proxemics-Net: Automatic Proxemics Recognition in Images](https://link.springer.com/chapter/10.1007/978-3-031-36616-1_32).

## Abstract

![Touch codes in Proxemics](img/proxemicsImg.png)

**Figure 1: Touch codes in Proxemics.** Images showing the six specific "touch codes" that were studied in this work.

Proxemics is a branch of anthropology that studies how humans use personal space as a means of nonverbal communication; that is, it studies how people interact. Due to the presence of physical contact between people, in the problem of proxemics recognition in images, we have to deal with occlusions and ambiguities, which complicates the process of recognition. Several papers have proposed different methods and models to solve this problem in recent years. Over the last few years, the rapid advancement of powerful Deep Learning techniques has resulted in novel methods and approaches. So, we propose Proxemics-Net, a new model that allows us to study the performance of two state-of-the-art deep learning architectures, ConvNeXt and Visual Transformers (as backbones) on the problem of classifying different types of proxemics on still images. Experiments on the existing Proxemics dataset show that these deep learning models do help favorably in the problem of proxemics recognition since we considerably outperformed the existing state of the art, with the ConvNeXt architecture being the best-performing backbone.


![Our Proxemics-Net model](img/Proxemics-Net.png)

**Figure 2: Our Proxemics-Net model.** It consists of the individual branches of each person (`p0_branch` and `p1_branch`) (blue) and the `pair branch` (red) as input. All branches consist of the same type of backbone (ConvNeXt or ViT). The outputs of these 3 branches are merged in a concatenation layer and passed through a fully connected layer that predicts the proxemic classes of the input samples.


## 1. File Structure

- `base_model_main/`: Main directory for the base model.
- `dataset/`: Directory containing the code necessary for dataset preprocessing.
- `test/`: Directory containing code and resources related to model testing.
- `train/`: Directory containing code and resources related to model training.
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

- Pre-trained Base model: [Download here](https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth) (350MB)
- Pre-trained Large model: [Download here](https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth) (800MB)

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

## References
If you find Proxemics-Net useful in your work, please consider citing the following BibTeX entry:
```bibtex
@InProceedings{jimenezVelasco2023,
   author = "Jiménez, I. and Muñoz, R. and Marín, M. J.",
   title = "Proxemics-Net: Automatic Proxemics Recognition in Images",
   booktitle = "Pattern Recogn. Image Anal.",
   year = "2023",
   pages = "402-413",
   note= "IbPRIA 2023",
   doi = "10.1007/978-3-031-36616-1_32"
}

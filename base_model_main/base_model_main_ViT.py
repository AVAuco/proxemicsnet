# ====================================================================
# MAIN : Train and test a BASE MODEL with ViT
# ====================================================================

import numpy as np
import argparse
import scipy.io
import cv2
import os
import sys
import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
import deepdish as dd
import pathlib 

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


sys.path.append("..")
sys.path.insert(0, '../test')
sys.path.insert(0, '../test')

from train.model import get_basemodel_ViT_newInput
from test import evaluateAP
from train.load_dataset_train_test import  get_partition_convNext
from train.datagenerator import *
from transformers import TFViTForImageClassification

import wandb
from wandb.keras import WandbCallback


def parse_args():
    parser = argparse.ArgumentParser(description='Training and testing script.')

   
    parser.add_argument('--b',type=int,  help='Size of each batch', required=False, default=6)
    parser.add_argument('--e', type=int,  help='Number of epochs', required=False, default=25)
    parser.add_argument('--lr', type=float,  help='lrate', required=False, default=0.01)
    parser.add_argument('--o',type=str,  help='optimizer', required=False, default="SGD")
    parser.add_argument('--g', type=float,  help='GPU rate', required=False, default=1)
    parser.add_argument('--set', type=int,  help='Set (1 or 2)', required=False, default=1)
    parser.add_argument('--onlyPairRGB',action='store_true',help='Only context brach of RGB model',default=False)
    parser.add_argument('--nlayersFreeze', type=int,  help='n layers frozen', choices=[0,2,4,6,8,10], required=False, default=0)

    #PATHS
    parser.add_argument('--datasetDIR',type=str,  help='Main Dir where dataset is located', required=True)
    parser.add_argument('--outModelsDIR',type=str,  help='Dir where model will be saved', required=True)

    return parser.parse_args()



#====================================================================================

if __name__ == '__main__':



    #Parameters read from console
    args = parse_args()
    #print(args)
    # ===========================================
    # Hyperparameters
    # ===========================================
    verbose=1
    batchsize = args.b
    nepochs = args.e
    gpuRate = args.g
    lr = args.lr
    optimizer = args.o
    useSet = args.set

    augmentation=True

    onlyPairRGB=args.onlyPairRGB
    nlayersFreeze=args.nlayersFreeze    
    
    datasetDir=args.datasetDIR
    outdir=args.outModelsDIR

    zipPath=os.path.join(datasetDir,'images/recortes.zip')

    # ===========================================
    # Model name
    # ===========================================
    if augmentation == True:
      aug=1
    else:
      aug=0

    typeImg='RGB'

    modelname = "Model_aug{:d}_bs{:d}_set{:d}_lr{:1.5f}_o{}_fr{:d}".format(aug,batchsize, useSet,lr, optimizer,nlayersFreeze)
    if onlyPairRGB:
      groupname=typeImg+'_onlypair_ViT'
    else:
      groupname=typeImg+'_p0p1pair_ViT'

    print(groupname)

    model_filepath = os.path.join(outdir,typeImg, groupname, modelname)
    
    #Directory where the model will be saved
    print("* The results will be saved to: "+model_filepath+"\n")
    sys.stdout.flush()

    # ===========================================
    # WANDB Parameters
    # ===========================================
    id='id_'+ groupname+ '_'+ modelname
    wandb.init(project="proxemics-convNext", group=groupname, name=modelname, id=id, config = {
      "learning_rate": lr,
      "epochs": nepochs,
      "batch_size": batchsize,
      "optimizer":optimizer,
      "set":useSet
    })

    
    
    # ===========================================
    # USE GPU
    # ===========================================
    #CHANGE ME
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpuRate
    tf.executing_eagerly()
    graph = tf.Graph()
    graph.as_default()
    session = tf.compat.v1.Session(graph=graph, config=config)
    session.as_default()



    # ===========================================
    # Dataset
    # ===========================================
    print('* Loading data')
    import json
    jsonfile=os.path.join(datasetDir,'labels_6classes_pair.json')
    with open(jsonfile) as file:
        dataset = json.load(file)

    #The information of all samples is stored in the ''all samples'' list.
    allSamples=[]
    imagenamelist=sorted(list(dataset.keys()))
    for image in imagenamelist:
      for pair in dataset[image]['proxemics'].keys():
        #print(image, pair)      # image ='0001.jpg'  / pair='p0-p1'
        
        label= dataset[image]['proxemics'][pair]
        p0=pair[0:2]
        p1=pair[3:]
        muestra=[image[:-4],p0,p1,label]
        #print(muestra)
        allSamples.append(muestra)


    npairs = len(allSamples)

    #We generate the data sets according to the division proposed by the authors
    trainImg,valImg,testImg=get_partition_convNext(useSet)

    #train samples
    trainIdx = []
    valIdx = []
    testIdx = []

    for idx in range(0,len(allSamples)):
      imgname = int(allSamples[idx][0])
      if imgname in trainImg:
        trainIdx.append(idx)
      elif imgname in valImg:
        valIdx.append(idx)
      else:
        testIdx.append(idx)


    # set partitions
    partition = {}
    partition['train'] = trainIdx #range(0,nsamples_train)       # IDs
    partition['validation'] = valIdx #range(nsamples_train, total_samples)
    partition['test'] = testIdx #range(nsamples_train, total_samples)
        

    # ===========================================
    #DataGenerators
    # ===========================================
    params = {
              'batch_size': batchsize,
              'shuffle': True,
              'augmentation': augmentation,
              'zipPath' : zipPath,
              'isTransformer' : True,
              'typeImg' : typeImg,
              'onlyPairRGB': onlyPairRGB
              }
    #DataGenerator for training
    training_generator = DataGenerator( partition['train'], allSamples, **params)
    partition['train_val'] = partition['train'] + partition['validation']
    training_validation_generator = DataGenerator( partition['train_val'], allSamples, **params)
    #DataGenerator for validation
    paramsVal= params
    paramsVal['shuffle'] = False
    paramsVal['augmentation'] = False
    paramsVal['isTest'] = True
    validation_generator =DataGenerator (partition['validation'], allSamples, **paramsVal)

    print("\n- Data generators are ready!\n")

    sys.stdout.flush()


    # ===========================================
    # MODEL
    # ===========================================
    if os.path.exists(os.path.join(model_filepath,'checkpoint')):
      print("INFO: loaded best model so far.")
      model = keras.models.load_model(os.path.join(model_filepath,'checkpoint'),custom_objects={'TFViTForImageClassification':TFViTForImageClassification }) 
    else:
      pathlib.Path(model_filepath).mkdir(parents=True, exist_ok=True)
      model = get_basemodel_ViT_newInput(lr,optimizer)

    #print(model.summary(expand_nested=True))
    print(model.summary())
    


    # ===========================================
    # TRAINING
    # ===========================================
    print("*** Starting training ***")
    metric='val_auc'
    checkpoint_cb = keras.callbacks.ModelCheckpoint(os.path.join(model_filepath,'checkpoint'), monitor=metric, save_format = "tf",verbose=1, save_best_only=True, save_freq="epoch", mode='max')
    #early_stopping_cb = keras.callbacks.EarlyStopping(monitor=metric, patience=8, verbose=1, min_delta=1e-4,restore_best_weights=True)
    reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(monitor=metric, factor=0.1,patience=4, verbose=1, min_delta=1e-4)
    #callbacks = [checkpoint_cb, early_stopping_cb, reduce_lr_cb]
    #callbacks = [ early_stopping_cb, reduce_lr_cb,WandbCallback()]
    #callbacks = [ checkpoint_cb, reduce_lr_cb]
    callbacks = [  reduce_lr_cb, checkpoint_cb, WandbCallback(save_model=False, monitor=metric)  ]

    # Go!
    nbatches = training_generator.__len__()
    history = model.fit(training_generator, epochs = nepochs, callbacks=callbacks, verbose=1, steps_per_epoch=nbatches, validation_data=validation_generator)
    nbatches = training_validation_generator.__len__()
    history = model.fit(training_validation_generator, epochs = 2, verbose=1, steps_per_epoch=nbatches) 

    print("*** End of training ***")
    
    # Save model
    model.save(model_filepath, save_format = "tf")


    # ===========================================
    # TEST
    # ===========================================
    #DataGenerator for test
    paramsTest = params
    paramsTest['shuffle'] = False
    paramsTest['augmentation'] = False
    paramsTest['isTest'] = True
    paramsTest['batch_size'] = len(partition['test'])

    test_generator =DataGenerator(partition['test'], allSamples, **paramsTest)
    X_test, y_test= test_generator.__getitem__(0)

    # 7. Evaluate model
    AP=evaluateAP(model,X_test,y_test)
    #Print AP results
    print()
    print("- AP results:")
    print(' HAND - HAND : ', AP['HAND_HAND'])
    print(' HAND - SHOULDER : ', AP['HAND_SHOULDER'])
    print(' SHOULDER - SHOULDER : ', AP['SHOULDER_SHOULDER'])
    print(' HAND - TORSO : ', AP['HAND_TORSO'])
    print(' HAND - ELBOW : ',AP['HAND_ELBOW'])
    print(' ELBOW - SHOULDER : ', AP['ELBOW_SHOULDER'])

    print()
    print("- mAP : " , AP['mAP'])

    # 8. Save results
    testFile = os.path.join(model_filepath,'best_keras_model_results.h5')
    dd.io.save(testFile, AP)

    wandb.log({"AP": AP})

    # 9. result per image
    testIdx=[]
    y_pred_img=[]
    y_test_img=[]
    for img in testImg:
      testIdx=[]
      for pix in range(0,len(allSamples)):				#Cogemos todas las muestras de ese video
        if  int(allSamples[pix][0]) == img:
          testIdx.append(pix)

      # Datasets
      partition = {}
      partition['test'] = testIdx
      paramsTest['batch_size'] = len(testIdx)

      

      test_generator = DataGenerator( partition['test'], allSamples, **paramsTest)
      X_test, y_test= test_generator.__getitem__(0)
      pred= model.predict(X_test)
      if len(pred) > 1:
        #print(pred)
        pred_per_img=pred.max(axis=0)
        test_per_img=y_test.max(axis=0)
        y_pred_img.append(pred_per_img.tolist())
        y_test_img.append(test_per_img.tolist())
      else:
        y_pred_img.append(pred[0].tolist())
        y_test_img.append(y_test[0].tolist())

    AP=evaluateAP(model,y_pred_img,y_test_img,False)

    #Print AP results
    print("- AP results CHECKPOINTS:")
    print(' HAND - HAND : ', AP['HAND_HAND'])
    print(' HAND - SHOULDER : ', AP['HAND_SHOULDER'])
    print(' SHOULDER - SHOULDER : ', AP['SHOULDER_SHOULDER'])
    print(' HAND - TORSO : ', AP['HAND_TORSO'])
    print(' HAND - ELBOW : ',AP['HAND_ELBOW'])
    print(' ELBOW - SHOULDER : ', AP['ELBOW_SHOULDER'])

    print("- mAP : " , AP['mAP'])

    testFile = os.path.join(model_filepath,'best_keras_model_results_chkt_perImage.h5')
    dd.io.save(testFile, AP)
  
    wandb.log({"AP_chkt_perImage": AP})



    wandb.finish()



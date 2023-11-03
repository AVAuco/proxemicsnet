#Script containing the necessary functions to generate any model(base_model, with pose)
# FUNCTIONS:
# -get_basemodel
# -get_posemodel 


# Required libraries
import numpy as np
import scipy.io
import cv2
import os
import copy

import tensorflow as tf
import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten,Input,Dropout, LayerNormalization
from tensorflow.keras.layers.experimental import preprocessing


from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import optimizers,layers

from transformers import TFViTForImageClassification


# Get the base model
def get_basemodel(lr,optimizador):
  # 1. Input layer
  input_tensor = Input(shape=(224, 224, 3),name='input')

  # 2. Pre-trained base model VGG16
  base_model = VGG16(include_top=True, input_tensor=input_tensor,weights='imagenet')

  # 3. Remove the output layer
  model= Sequential(base_model.layers[:-1])

  # 4. Add output layer with sigmoid activation function
  model.add(Dense(6, activation="sigmoid",name='output'))

  # 5. We generate and compile the model
  finalModel= Model(inputs=base_model.input, outputs=model.output)

  if optimizador=='Adam':
    opt = optimizers.Adam(learning_rate=lr)
  else:
    opt = optimizers.SGD(learning_rate=lr)

  finalModel.compile(loss='binary_crossentropy', optimizer=opt, metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC(num_thresholds=3),tf.keras.metrics.Precision()])

  return finalModel



############################ POSE
#Pose branch for vector
def getPose_branch_vector( shapePoseRepresentation=652288,dropoutval=0.5, batchnorm=False, usel2=False):
    the_input_shape = (shapePoseRepresentation,)

    mapBranch = Sequential(name="poseMapBranch")
    mapBranch.add(Dense(256, input_shape=the_input_shape, activation='relu'))
    mapBranch.add(Dense(128, activation='relu'))
    mapBranch.add(Dense(64, activation='relu'))
    mapBranch.add(Dense(32, activation='relu'))


    mapBranch.add(Dropout(dropoutval, name="ptop_dropout"))


    return mapBranch


#Pose branch with CONV3D for heatmaps
def getPose_branch_heatmaps_3D(shapePoseRepresentation=13 , dropoutval=0.5):
    # Some parameters
    the_input_shape= (shapePoseRepresentation, 224, 224, 1)
    mapBranch = Sequential(name="poseMapBranch")
    key="p"

    if (shapePoseRepresentation==13):
        mapBranch.add(layers.Conv3D(16, (3,5,5), strides=(1, 2, 2), padding='valid', data_format='channels_last',activation='relu', input_shape=the_input_shape, name=(key+"conv3d_1")))

        mapBranch.add(layers.Conv3D(24, (3,3,3), strides=(1, 2, 2), padding='valid', data_format='channels_last', activation='relu', name=(key+"conv3d_2")))

    else:
        mapBranch.add(layers.Conv3D(16, (2,5,5), strides=(1, 2, 2), padding='valid', data_format='channels_last',activation='relu', input_shape=the_input_shape, name=(key+"conv3d_1")))
        
        mapBranch.add(layers.Conv3D(24, (1,3,3), strides=(1, 2, 2), padding='valid', data_format='channels_last',activation='relu', name=(key+"conv3d_2")))


    mapBranch.add(layers.Conv3D(32, (1,3,3), strides=(1, 2, 2), padding='valid', data_format='channels_last',
                           activation='relu', name=(key+"conv3d_3")))

    mapBranch.add(layers.Conv3D(12, (1,6,6), strides=(1, 1, 1), padding='valid', data_format='channels_last',
                           activation='relu', name=(key+"conv3d_4")))

    mapBranch.add(layers.Dropout(dropoutval, name=(key+"top_dropout")))

    mapBranch.add(layers.Flatten(name=(key+"flat")))

 
    return mapBranch

#Pose branch with CONV3D for heatmaps 
def getPose_branch_heatmaps_3D_RANDOM(shapePoseRepresentation=13 , dropoutval=0.5):
    # Some parameters
    the_input_shape= (2, 224, 224,1)
    mapBranch = Sequential([
    #tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal",input_shape=( 224, 224)),
    #preprocessing.RandomBrightnes(factor=0.2),
    preprocessing.RandomContrast(factor=0.4)],name="poseMapBranch")
    #mapBranch = Sequential([data_augmentation],name="poseMapBranch")
    key="p"

    if (shapePoseRepresentation==13):
        mapBranch.add(layers.Conv3D(16, (3,5,5), strides=(1, 2, 2), padding='valid', data_format='channels_last',activation='relu', input_shape=the_input_shape, name=(key+"conv3d_1")))

        mapBranch.add(layers.Conv3D(24, (3,3,3), strides=(1, 2, 2), padding='valid', data_format='channels_last', activation='relu', name=(key+"conv3d_2")))

    else:

        mapBranch.add(layers.Conv3D(16, (2,5,5), strides=(1, 2, 2), padding='valid', data_format='channels_last',activation='relu', input_shape=(2,224,224,1), name=(key+"conv3d_1")))
        
        mapBranch.add(layers.Conv3D(24, (1,3,3), strides=(1, 2, 2), padding='valid', data_format='channels_last',activation='relu', name=(key+"conv3d_2")))


    mapBranch.add(layers.Conv3D(32, (1,3,3), strides=(1, 2, 2), padding='valid', data_format='channels_last',
                           activation='relu', name=(key+"conv3d_3")))

    mapBranch.add(layers.Conv3D(12, (1,6,6), strides=(1, 1, 1), padding='valid', data_format='channels_last',
                           activation='relu', name=(key+"conv3d_4")))

    mapBranch.add(layers.Dropout(dropoutval, name=(key+"top_dropout")))

    mapBranch.add(layers.Flatten(name=(key+"flat")))

 
    return mapBranch



# Get the pose model (BASELINE +POSE)
'''
def get_posemodel(lr,optimizador,typeRepresentation='heatmap',shapePoseRepresentation=13 ):
  # 1. Input layer
  rgbinput = Input(shape=(224, 224, 3),name='rgbInput')
  
  # 2. Pre-trained base model VGG16
  base_model = VGG16(include_top=True, input_tensor=rgbinput,weights='imagenet')

  # 3. Remove the output layer
  base_model_encoding = Sequential(base_model.layers[:-3])
  
  # 4. Add pose branch
  if typeRepresentation=='heatmap':    #HEATMAPS
    poseMapinput=Input(shape=(shapePoseRepresentation,224,224,1),name='poseInput')
    poseMapBranch = getPose_branch_heatmaps_3D(shapePoseRepresentation)
  elif typeRepresentation=='vector':                                #VECTOR
    poseMapinput=Input(shape=(shapePoseRepresentation,),name='poseInput')
    poseMapBranch = getPose_branch_vector(shapePoseRepresentation)
  else:
    print('ERROR : Representation does not exist')
    
   
  poseMap_encoding = poseMapBranch(poseMapinput)

  # 5. Concatenate branches
  the_concats = [base_model_encoding.output, poseMap_encoding]
  the_inputs = [rgbinput, poseMapinput]

  concat =layers.concatenate(the_concats, name="concat_encods")

  # 6. Add intermediate layers
  x = layers.Dense(4096, activation='relu', name='fc_1')(concat)
  x = layers.Dropout(0.5)(x)
  x = layers.Dense(4096, activation='relu', name='fc_2')(x)

  drp2 = layers.Dropout(0.5, name="top_dropout")(x)

  # 7. Add output layer with sigmoid activation function
  proxemicsoutput = layers.Dense(6, activation='sigmoid', name='output')(drp2)
  #model.add(Dense(6, activation="sigmoid",name='output'))

  # 8. We generate and compile the model
  finalModel = Model(inputs=the_inputs, outputs=proxemicsoutput, name="Proxemics_pose")
 
  # 9. Optimizer
  if optimizador=='Adam':
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
  else:
    opt = tf.keras.optimizers.SGD(learning_rate=lr)

  #10. Compile model
  finalModel.compile(loss='binary_crossentropy', optimizer=opt, metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC(num_thresholds=3),tf.keras.metrics.Precision()])
  
  return finalModel
'''

# Get the pose model
def get_posemodel(lr,optimizador,typeRepresentation='heatmap',shapePoseRepresentation=13 ):
  if typeRepresentation=='heatmap':    #HEATMAPS
    poseMapinput=Input(shape=(shapePoseRepresentation,224,224,1),name='poseInput')
    poseMapBranch = getPose_branch_heatmaps_3D_RANDOM(shapePoseRepresentation)
  elif typeRepresentation=='vector':                                #VECTOR
    poseMapinput=Input(shape=(shapePoseRepresentation,),name='poseInput')
    poseMapBranch = getPose_branch_vector(shapePoseRepresentation)
  else:
    print('ERROR : Representation does not exist')
    
  
  poseMap_encoding = poseMapBranch(poseMapinput)


  # 6. Add intermediate layers
  x = layers.Dense(4096, activation='relu', name='fc_1')(poseMap_encoding)
  x = layers.Dropout(0.5)(x)
  x = layers.Dense(4096, activation='relu', name='fc_2')(x)

  drp2 = layers.Dropout(0.5, name="top_dropout")(x)

  # 7. Add output layer with sigmoid activation function
  proxemicsoutput = layers.Dense(6, activation='sigmoid', name='output')(drp2)
  #model.add(Dense(6, activation="sigmoid",name='output'))

  # 8. We generate and compile the model
  finalModel = Model(inputs=poseMapinput, outputs=proxemicsoutput, name="Proxemics_pose")
 
  # 9. Optimizer
  if optimizador=='Adam':
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
  else:
    opt = tf.keras.optimizers.SGD(learning_rate=lr)

  #10. Compile model
  finalModel.compile(loss='binary_crossentropy', optimizer=opt, metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC(num_thresholds=3),tf.keras.metrics.Precision()])
  
  return finalModel


############################################################################### TRANSFORMER ########################################################################



def get_model_ViT(lr,optimizador):
        
        #transformer = TFViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k",num_labels=6, output_hidden_states=True)
        transformer = TFViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k",num_labels=6)


        inputs = tf.keras.layers.Input(shape=(3,224,224), dtype=tf.float32,name='rgbInput' )
        model = transformer(inputs)

        proxemicsoutput = tf.keras.layers.Dense(6, activation='sigmoid',name='Output')(model[0])

        finalModel = tf.keras.models.Model(inputs = inputs, outputs = proxemicsoutput,name="ProxemicsBase")



        # 9. Optimizer
        if optimizador=='Adam':
          opt = tf.keras.optimizers.Adam(learning_rate=lr)
        else:
          opt = tf.keras.optimizers.SGD(learning_rate=lr)

        #10. Compile model
        finalModel.compile(loss='binary_crossentropy', optimizer=opt, metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC(num_thresholds=3),tf.keras.metrics.Precision()])
        
        return finalModel



def get_posemodel_ViT(lr,optimizador,typeRepresentation='heatmap',shapePoseRepresentation=13):
        
        #transformer = TFViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k",num_labels=6, output_hidden_states=True)
        transformer = TFViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k",num_labels=6,name='ViT_Transformer')


        rgbinput = tf.keras.layers.Input(shape=(3,224,224), dtype=tf.float32,name='rgbInput' )
        model = transformer(rgbinput)

        # 4. Add pose branch
        if typeRepresentation=='heatmap':    #HEATMAPS
          poseMapinput=Input(shape=(shapePoseRepresentation,224,224,1), name='poseInput')
          poseMapBranch = getPose_branch_heatmaps_3D(shapePoseRepresentation)
        elif typeRepresentation=='vector':                                #VECTOR
          poseMapinput=Input(shape=(shapePoseRepresentation,), name='poseInput')
          poseMapBranch = getPose_branch_vector(shapePoseRepresentation)
        else:
          print('ERROR : Representation does not exist')

        poseMap_encoding = poseMapBranch(poseMapinput)

        # 5. Concatenate branches
        the_concats = [model[0], poseMap_encoding]
        the_inputs = [rgbinput, poseMapinput]


        concat =layers.concatenate(the_concats, name="concat_encods")

        # 6. Add intermediate layers
        x = layers.Dense(4096, activation='relu', name='fc_1')(concat)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(4096, activation='relu', name='fc_2')(x)

        drp2 = layers.Dropout(0.5, name="top_dropout")(x)

        proxemicsoutput = tf.keras.layers.Dense(6, activation='sigmoid',name='Output')(drp2)

        finalModel = tf.keras.models.Model(inputs = the_inputs, outputs = proxemicsoutput,name="Proxemics_pose_Vit")



        # 9. Optimizer
        if optimizador=='Adam':
          opt = tf.keras.optimizers.Adam(learning_rate=lr)
        else:
          opt = tf.keras.optimizers.SGD(learning_rate=lr)

        #10. Compile model
        finalModel.compile(loss='binary_crossentropy', optimizer=opt, metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC(num_thresholds=3),tf.keras.metrics.Precision()])
        
        return finalModel


def get_basemodel_ViT_newInput(lr,optimizador):

        
  #transformer = TFViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k",num_labels=6, output_hidden_states=True)
  transformer0 = TFViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k",num_labels=6,name='ViT_Transformer_p0')
  rgbinput0 = tf.keras.layers.Input(shape=(3,224,224), dtype=tf.float32,name='rgbInput_p0' )
  base_model0 = transformer0(rgbinput0)

  transformer1 = TFViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k",num_labels=6,name='ViT_Transformer_p1')
  rgbinput1 = tf.keras.layers.Input(shape=(3,224,224), dtype=tf.float32,name='rgbInput_p1' )
  base_model1 = transformer1(rgbinput1)

  transformer_pair = TFViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k",num_labels=6,name='ViT_Transformer_pair')
  rgbinput_pair = tf.keras.layers.Input(shape=(3,224,224), dtype=tf.float32,name='rgbInput_pair' )
  base_model_pair = transformer_pair(rgbinput_pair)



  # 5. Concatenate branches
  the_concats = [base_model0[0], base_model1[0],base_model_pair[0]]
  the_inputs = [rgbinput0, rgbinput1, rgbinput_pair]


  concat =layers.concatenate(the_concats, name="concat_encods")

  # 6. Add intermediate layers
  x = layers.Dense(4096, activation='relu', name='fc_1')(concat)
  x = layers.Dropout(0.5)(x)
  x = layers.Dense(4096, activation='relu', name='fc_2')(x)

  drp2 = layers.Dropout(0.5, name="top_dropout")(x)

  proxemicsoutput = tf.keras.layers.Dense(6, activation='sigmoid',name='Output')(drp2)

  finalModel = tf.keras.models.Model(inputs = the_inputs, outputs = proxemicsoutput,name="Proxemics_Vit_newInput")



  # 9. Optimizer
  if optimizador=='Adam':
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
  else:
    opt = tf.keras.optimizers.SGD(learning_rate=lr)

  #10. Compile model
  finalModel.compile(loss='binary_crossentropy', optimizer=opt, metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC(num_thresholds=3),tf.keras.metrics.Precision()])
  
  return finalModel



############################################################################### ConvNext ########################################################################
def generate(base_model,nlayersFreeze,name):
   finalModel=Sequential(name=name)
   i=0
   for layer in base_model.layers[1:-1]:
      if i < nlayersFreeze:
        layer.trainable=False
      
      finalModel.add(layer)
      i=i+1
         

   return finalModel


# Get the convnext base model
def get_basemodel_convNext(model_gcs_path,lr,optimizador, typeImg, onlyPairRGB, onlyPairPose, nlayersFreeze):
  #ndef get_basemodel_convNext(model_gcs_path,lr,optimizador, typeImg, onlyPairRGB, onlyPairPose, nlayersFreeze,smo):
  # 2. Pre-trained base model 
  
  #mirrored_strategy = tf.distribute.MirroredStrategy()
  #print(mirrored_strategy)
  
  the_concats=[]
  the_inputs=[]
  #with mirrored_strategy.scope(): 
  if 'RGB' in typeImg:
    if onlyPairRGB==False:
      base_model_convnext =  tf.keras.models.load_model(model_gcs_path,compile=False )

      rgbinputp0 = tf.keras.layers.Input(shape=(224,224,3), dtype=tf.float32,name='rgbinputp0' )
      rgbinputp1 = tf.keras.layers.Input(shape=(224,224,3), dtype=tf.float32,name='rgbinputp1' )
    
      base_model=generate(base_model_convnext,nlayersFreeze,"individualbranch")
      
      base_model0 = base_model(rgbinputp0)
      base_model1 = base_model(rgbinputp1)

      the_concats = [base_model0, base_model1]
      the_inputs = [rgbinputp0,rgbinputp1]

    base_model_pair =  tf.keras.models.load_model(model_gcs_path,compile=False)
    i=0
    for layer in base_model_pair.layers:
      layer._name = layer.name + str("_pair") 
      if i > 0:
        if i <= nlayersFreeze:
          layer.trainable=False
      i=i+1
        
        #layer.trainable=False

    the_concats.append(base_model_pair.layers[-2].output)
    the_inputs.append(base_model_pair.input)

  
  if 'Pose' in typeImg:
    if onlyPairPose==False:
      pose_model_convnext =  tf.keras.models.load_model(model_gcs_path,compile=False )

      pose_rgbinputp0 = tf.keras.layers.Input(shape=(224,224,3), dtype=tf.float32,name='pose_rgbinputp0' )
      pose_rgbinputp1 = tf.keras.layers.Input(shape=(224,224,3), dtype=tf.float32,name='pose_rgbinputp1' )

      pose_model=generate(pose_model_convnext,nlayersFreeze,"individualbranch_pose")
      
      pose_model0 = pose_model(pose_rgbinputp0)
      pose_model1 = pose_model(pose_rgbinputp1)

      the_concats.append(pose_model0)
      the_concats.append(pose_model1)
      the_inputs.append(pose_rgbinputp0)
      the_inputs.append(pose_rgbinputp1)



    pose_model_pair =  tf.keras.models.load_model(model_gcs_path,compile=False)
    i=0
    for layer in pose_model_pair.layers:
      layer._name = layer.name + str("_pair_pose") 
      if i > 0:
        if i <= nlayersFreeze:
          layer.trainable=False
      i=i+1
    
    the_concats.append(pose_model_pair.layers[-2].output)
    the_inputs.append(pose_model_pair.input)          
  

  if typeImg != 'RGB_Pose':
      if onlyPairRGB==True or onlyPairPose==True:
        concat=the_concats[0]
        the_inputs=the_inputs[0]
      else:
        concat =layers.concatenate(the_concats, name="concat_encods",axis=-1)
        concat = LayerNormalization()(concat)
  else:    
    concat =layers.concatenate(the_concats, name="concat_encods",axis=-1)
    concat = LayerNormalization()(concat)

  



  # 6. Add intermediate layers
  #axis=1
  
  x = layers.Dense(4096, activation='relu', name='fc_1')(concat)
  x = layers.Dropout(0.5)(x)
  x = layers.Dense(4096, activation='relu', name='fc_2')(x)

  drp2 = layers.Dropout(0.5, name="top_dropout")(x)

  proxemicsoutput = layers.Dense(6, activation='sigmoid', name='output')(drp2)

  

  # 5. We generate and compile the model
  finalModel= Model(inputs=the_inputs, outputs=proxemicsoutput)


  # 9. Optimizer
  if optimizador=='Adam':
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
  if optimizador=='AdamW':
    opt = tf.keras.optimizers.experimental.AdamW(learning_rate=lr)
  else:
    opt = tf.keras.optimizers.SGD(learning_rate=lr)

  #10. Compile model
  finalModel.compile( loss='binary_crossentropy', optimizer=opt, metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC(num_thresholds=3),tf.keras.metrics.Precision()])
  #finalModel.compile( loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=smo, name='binary_crossentropy'), optimizer=opt, metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC(num_thresholds=3),tf.keras.metrics.Precision()])

  return finalModel

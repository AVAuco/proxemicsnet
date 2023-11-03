#Script containing the necessary functions to prepare the dataset to train and test
# FUNCTIONS:
# -get_dataset
# -get_train_test_partition_author
# -get_train_test_partition_author_pose
# -feature_extractor_ViT
# -get_dataset_ViT
# -get_train_test_partition_author_ViT
# -get_train_test_partition_author_pose_ViT

import numpy as np
import scipy.io
import cv2
import os
import sys
import deepdish as dd
from PIL import Image

from transformers import AutoFeatureExtractor

# Get the dataset
def get_dataset(filename,datasetDir):   #datasetDir=/Proxemics/
  # 1. We load the .mat file containing the path to the images and their corresponding labels.
  data = scipy.io.loadmat(filename)
  
  nSamples=len(data['Proxemics'])

  # 2. Numpy vectors for saving images and labels
  X=np.zeros((nSamples,224,224,3))
  y=np.zeros((nSamples,6))

  # 3. We scroll through each sample of the read file
  #Check operative system to load image correctly
  my_os=sys.platform
  print("System OS : ",my_os,'\n')
  for sample in range(0, nSamples):
    # 2.1 Load image
    path=data['Proxemics'][sample][0][0]
    imgPath = os.path.join(datasetDir,path)

    if my_os =='linux':
        img = cv2.imread(imgPath)
    else:
        #Windows11
        imgPath = imgPath.replace('/','\\')
        img = cv2.imread(imgPath)

    # We save image in the vector of samples X 
    X[sample,]=np.array(img) 

    # 2.2 We save label in the vector of labels y 
    y[sample,]=data['Proxemics'][sample][1][0]

  return X, y



def get_train_test_partition_author(X,y,set):
  # 1. Indexes of the images that will correspond to train and test
  #trainfrs = [1:300 590:889]; 
	#testfrs = [301:589 890:1178];
  p1=[]
  p2=[]
  for i in range(0, 1178):
    if i in range(0,300) or i in range(589,889):
      p1.append(i)
    if i in range(300,589) or i in range(889,1178):
      p2.append(i)
  
  # 2. Set 
  if set==1:
    train_val=p1
    test=p2
  else:
    train_val=p2
    test=p1

  # TRAIN/VALIDATION/TEST
  train=[]
  validation=[]

  for i in range(0, len(train_val)):
    if i in [1,11,21,31,41,51,61,71,81,91,101,111,121,131,141,151,161,171,181,191,201,211,221,231,241,251,261,271,281,291,301,311,321,331,341,351,361,371,381,391,401,411,421,431,441,451,461,471,481,491,501,511,521,531,541,551,561,571]:
      validation.append(train_val[i])
    else:
      train.append(train_val[i])

  
  
  # 3. Number of idx for each set
  nSamplesTrain=len(train)
  nSamplesvalidation=len(validation)
  nSamplesTest=len(test)


  # 4. Numpy vectors for saving images and labels
  X_train=np.zeros((nSamplesTrain,224,224,3))
  X_val=np.zeros((nSamplesvalidation,224,224,3))
  X_test=np.zeros((nSamplesTest,224,224,3))
  y_train=np.zeros((nSamplesTrain,6))
  y_val=np.zeros((nSamplesvalidation,6))
  y_test=np.zeros((nSamplesTest,6))

  # 5. We scroll through the vector with the indexes of the images that will go in X_train
  sample=0
  for idx in train:
    # We save the image and label that corresponds to this index.
    X_train[sample,]=X[idx] 
    y_train[sample,]=y[idx]

    sample=sample+1
  
  #6. We scroll through the vector with the indexes of the images that will go in X_val
  sample=0
  for idx in validation:
    # We save the image and label that corresponds to this index.
    X_val[sample,]=X[idx] 
    y_val[sample,]=y[idx]

    sample=sample+1
  
  #7. We scroll through the vector with the indexes of the images that will go in X_test
  sample=0
  for idx in test:
    # We save the image and label that corresponds to this index.
    X_test[sample,]=X[idx] 
    y_test[sample,]=y[idx]

    sample=sample+1

  return X_train, y_train, X_val, y_val, X_test, y_test





def get_train_test_partition_author_pose(X,y,set,posefilePath,typeRepresentation='heatmap',shapePoseRepresentation=13):
  # 1. Indexes of the images that will correspond to train and test
  #trainfrs = [1:300 590:889]; 
	#testfrs = [301:589 890:1178];
  p1=[]
  p2=[]
  for i in range(0, 1178):
    if i in range(0,300) or i in range(589,889):
      p1.append(i)
    if i in range(300,589) or i in range(889,1178):
      p2.append(i)
  
  # 2. Set 
  if set==1:
    train_val=p1
    test=p2
  else:
    train_val=p2
    test=p1

  # TRAIN/VALIDATION/TEST
  train=[]
  validation=[]

  for i in range(0, len(train_val)):
    if i in [1,11,21,31,41,51,61,71,81,91,101,111,121,131,141,151,161,171,181,191,201,211,221,231,241,251,261,271,281,291,301,311,321,331,341,351,361,371,381,391,401,411,421,431,441,451,461,471,481,491,501,511,521,531,541,551,561,571]:
      validation.append(train_val[i])
    else:
      train.append(train_val[i])

  
  
  # 3. Number of idx for each set
  nSamplesTrain=len(train)
  nSamplesvalidation=len(validation)
  nSamplesTest=len(test)


  # 4. Numpy vectors for saving images and labels
  X_train=[]
  X_val=[]
  X_test=[]

  X_train_img=np.zeros((nSamplesTrain,224,224,3))
  X_val_img=np.zeros((nSamplesvalidation,224,224,3))
  X_test_img=np.zeros((nSamplesTest,224,224,3))

  if (typeRepresentation=='heatmap'):
    X_train_pose=np.zeros((nSamplesTrain,shapePoseRepresentation,224,224))
    X_val_pose=np.zeros((nSamplesvalidation,shapePoseRepresentation,224,224))
    X_test_pose=np.zeros((nSamplesTest,shapePoseRepresentation,224,224))


  else:    
    X_train_pose=np.zeros((nSamplesTrain,shapePoseRepresentation))
    X_val_pose=np.zeros((nSamplesvalidation,shapePoseRepresentation))
    X_test_pose=np.zeros((nSamplesTest,shapePoseRepresentation))


  y_train=np.zeros((nSamplesTrain,6))
  y_val=np.zeros((nSamplesvalidation,6))
  y_test=np.zeros((nSamplesTest,6))

  # 5. We scroll through the vector with the indexes of the images that will go in X_train
  sample=0
  for idx in train:
    # We save the image and label that corresponds to this index.
    X_train_img[sample,]=X[idx] 
    y_train[sample,]=y[idx]

 
    #print(idx)
    if (idx+1) > 999 :
        key='img'+str(idx+1)
    elif (idx+1) > 99 and (idx+1) < 1000:
        key='img0'+str(idx+1)
    elif (idx+1) > 9 and (idx+1) < 100:
        key='img00'+str(idx+1)
    else:
        key='img000'+str(idx+1)

    #print(key)
    file=posefilePath+key+'.h5'
    print(file)
    X_train_pose[sample,]=dd.io.load(file)

    sample=sample+1

  # 5.1. Normalize from [0-255] to [0-1]
  X_train_img =X_train_img.astype('float32')  
  X_train_img /= 255.0
  
  X_train.append(X_train_img)
  X_train.append(X_train_pose)


  # 6. We scroll through the vector with the indexes of the images that will go in X_val
  sample=0
  for idx in validation:
    # We save the image and label that corresponds to this index.
    X_val_img[sample,]=X[idx] 
    y_val[sample,]=y[idx]

    #print(idx)
    if (idx+1) > 999 :
        key='img'+str(idx+1)
    elif (idx+1) > 99 and (idx+1) < 1000:
        key='img0'+str(idx+1)
    elif (idx+1) > 9 and (idx+1) < 100:
        key='img00'+str(idx+1)
    else:
        key='img000'+str(idx+1)

    #print(key)
    file=posefilePath+key+'.h5'
    print(file)
    X_val_pose[sample,]=dd.io.load(file)


    sample=sample+1

  # 6.1. Normalize from [0-255] to [0-1]
  X_val_img =X_val_img.astype('float32')  
  X_val_img /= 255.0
  
  X_val.append(X_val_img)
  X_val.append(X_val_pose)




  #7. We scroll through the vector with the indexes of the images that will go in X_test
  sample=0
  for idx in test:

    # We save the image and label that corresponds to this index.
    X_test_img[sample,]=X[idx] 
    y_test[sample,]=y[idx]


    #print(idx)
    if (idx+1) > 999 :
        key='img'+str(idx+1)
    elif (idx+1) > 99 and (idx+1) < 1000:
        key='img0'+str(idx+1)
    elif (idx+1) > 9 and (idx+1) < 100:
        key='img00'+str(idx+1)
    else:
        key='img000'+str(idx+1)
      
    #print(key)
    file=posefilePath+key+'.h5'
    print(file)
    X_test_pose[sample,]=dd.io.load(file)

    sample=sample+1
  
  # 7.1. Normalize from [0-255] to [0-1]
  X_test_img =X_test_img.astype('float32')  
  X_test_img /= 255.0

  X_test.append(X_test_img)
  X_test.append(X_test_pose)
  

  return X_train, y_train, X_val, y_val, X_test, y_test









############################################################################### TRANSFORMER ########################################################################
def feature_extractor_ViT(X):
  
  feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

  X_transformer=np.zeros((len(X),3,224,224))

  nsamples=0
  for i in X:
    X_transformer[nsamples]=feature_extractor(i, return_tensors='pt')['pixel_values']
    nsamples=nsamples+1

  return X_transformer


def get_dataset_ViT(filename,datasetDir):   #datasetDir=/Proxemics/
  # 1. We load the .mat file containing the path to the images and their corresponding labels.
  data = scipy.io.loadmat(filename)
  
  nSamples=len(data['Proxemics'])

  # 2. Numpy vectors for saving images and labels
  X=[]
  y=np.zeros((nSamples,6))

  # 3. We scroll through each sample of the read file
  #Check operative system to load image correctly
  my_os=sys.platform
  print("System OS : ",my_os,'\n')
  for sample in range(0, nSamples):
    # 2.1 Load image
    path=data['Proxemics'][sample][0][0]
    imgPath = os.path.join(datasetDir,path)

    if my_os =='linux':
        img = Image.open(imgPath)
    else:
        #Windows11
        imgPath = imgPath.replace('/','\\')
        img = Image.open(imgPath)

    # We save image in the vector of samples X 
    X.append(img.copy())
    img.close()

    # 2.2 We save label in the vector of labels y 
    y[sample,]=data['Proxemics'][sample][1][0]

  return X, y



def get_train_test_partition_author_ViT(X,y,set):
  # 1. Indexes of the images that will correspond to train and test
  #trainfrs = [1:300 590:889]; 
	#testfrs = [301:589 890:1178];
  p1=[]
  p2=[]
  for i in range(0, 1178):
    if i in range(0,300) or i in range(589,889):
      p1.append(i)
    if i in range(300,589) or i in range(889,1178):
      p2.append(i)
  
  # 2. Set 
  if set==1:
    train_val=p1
    test=p2
  else:
    train_val=p2
    test=p1

  # TRAIN/VALIDATION/TEST
  train=[]
  validation=[]

  for i in range(0, len(train_val)):
    if i in [1,11,21,31,41,51,61,71,81,91,101,111,121,131,141,151,161,171,181,191,201,211,221,231,241,251,261,271,281,291,301,311,321,331,341,351,361,371,381,391,401,411,421,431,441,451,461,471,481,491,501,511,521,531,541,551,561,571]:
      validation.append(train_val[i])
    else:
      train.append(train_val[i])

  
  
  # 3. Number of idx for each set
  nSamplesTrain=len(train)
  nSamplesvalidation=len(validation)
  nSamplesTest=len(test)


  # 4. Numpy vectors for saving images and labels
  X_train=[]
  X_val=[]
  X_test=[]
  y_train=np.zeros((nSamplesTrain,6))
  y_val=np.zeros((nSamplesvalidation,6))
  y_test=np.zeros((nSamplesTest,6))

  # 5. We scroll through the vector with the indexes of the images that will go in X_train
  sample=0
  for idx in train:
    # We save the image and label that corresponds to this index.
    X_train.append(X[idx]) 
    y_train[sample,]=y[idx]

    sample=sample+1
  
  #6. We scroll through the vector with the indexes of the images that will go in X_val
  sample=0
  for idx in validation:
    # We save the image and label that corresponds to this index.
    X_val.append(X[idx])
    y_val[sample,]=y[idx]

    sample=sample+1
  
  #7. We scroll through the vector with the indexes of the images that will go in X_test
  sample=0
  for idx in test:
    # We save the image and label that corresponds to this index.
    X_test.append(X[idx])
    y_test[sample,]=y[idx]

    sample=sample+1

  return X_train, y_train, X_val, y_val, X_test, y_test



def get_train_test_partition_author_pose_ViT(X,y,set,posefilePath,typeRepresentation='heatmap',shapePoseRepresentation=13):
  # 1. Indexes of the images that will correspond to train and test
  #trainfrs = [1:300 590:889]; 
	#testfrs = [301:589 890:1178];
  p1=[]
  p2=[]
  for i in range(0, 1178):
    if i in range(0,300) or i in range(589,889):
      p1.append(i)
    if i in range(300,589) or i in range(889,1178):
      p2.append(i)
  
  # 2. Set 
  if set==1:
    train_val=p1
    test=p2
  else:
    train_val=p2
    test=p1

  # TRAIN/VALIDATION/TEST
  train=[]
  validation=[]

  for i in range(0, len(train_val)):
    if i in [1,11,21,31,41,51,61,71,81,91,101,111,121,131,141,151,161,171,181,191,201,211,221,231,241,251,261,271,281,291,301,311,321,331,341,351,361,371,381,391,401,411,421,431,441,451,461,471,481,491,501,511,521,531,541,551,561,571]:
      validation.append(train_val[i])
    else:
      train.append(train_val[i])

  
  
  # 3. Number of idx for each set
  nSamplesTrain=len(train)
  nSamplesvalidation=len(validation)
  nSamplesTest=len(test)


  # 4. Numpy vectors for saving images and labels
  X_train=[]
  X_val=[]
  X_test=[]

  X_train_img=[]
  X_val_img=[]
  X_test_img=[]

  if (typeRepresentation=='heatmap'):
    X_train_pose=np.zeros((nSamplesTrain,shapePoseRepresentation,224,224))
    X_val_pose=np.zeros((nSamplesvalidation,shapePoseRepresentation,224,224))
    X_test_pose=np.zeros((nSamplesTest,shapePoseRepresentation,224,224))


  else:
    X_train_pose=np.zeros((nSamplesTrain,shapePoseRepresentation))
    X_val_pose=np.zeros((nSamplesvalidation,shapePoseRepresentation))
    X_test_pose=np.zeros((nSamplesTest,shapePoseRepresentation))


  y_train=np.zeros((nSamplesTrain,6))
  y_val=np.zeros((nSamplesvalidation,6))
  y_test=np.zeros((nSamplesTest,6))

  # 5. We scroll through the vector with the indexes of the images that will go in X_train
  sample=0
  for idx in train:
    # We save the image and label that corresponds to this index.
    X_train_img.append(X[idx]) 
    y_train[sample,]=y[idx]

 
    #print(idx)
    if (idx+1) > 999 :
        key='img'+str(idx+1)
    elif (idx+1) > 99 and (idx+1) < 1000:
        key='img0'+str(idx+1)
    elif (idx+1) > 9 and (idx+1) < 100:
        key='img00'+str(idx+1)
    else:
        key='img000'+str(idx+1)

    #print(key)
    file=posefilePath+key+'.h5'
    print(file)
    X_train_pose[sample,]=dd.io.load(file)

    sample=sample+1

  
  X_train.append(X_train_img)
  X_train.append(X_train_pose)


  # 6. We scroll through the vector with the indexes of the images that will go in X_val
  sample=0
  for idx in validation:
    # We save the image and label that corresponds to this index.
    X_val_img.append(X[idx]) 
    y_val[sample,]=y[idx]

    #print(idx)
    if (idx+1) > 999 :
        key='img'+str(idx+1)
    elif (idx+1) > 99 and (idx+1) < 1000:
        key='img0'+str(idx+1)
    elif (idx+1) > 9 and (idx+1) < 100:
        key='img00'+str(idx+1)
    else:
        key='img000'+str(idx+1)

    #print(key)
    file=posefilePath+key+'.h5'
    print(file)
    X_val_pose[sample,]=dd.io.load(file)


    sample=sample+1


  X_val.append(X_val_img)
  X_val.append(X_val_pose)




  #7. We scroll through the vector with the indexes of the images that will go in X_test
  sample=0
  for idx in test:

    # We save the image and label that corresponds to this index.
    X_test_img.append(X[idx]) 
    y_test[sample,]=y[idx]


    #print(idx)
    if (idx+1) > 999 :
        key='img'+str(idx+1)
    elif (idx+1) > 99 and (idx+1) < 1000:
        key='img0'+str(idx+1)
    elif (idx+1) > 9 and (idx+1) < 100:
        key='img00'+str(idx+1)
    else:
        key='img000'+str(idx+1)
      
    #print(key)
    file=posefilePath+key+'.h5'
    print(file)
    X_test_pose[sample,]=dd.io.load(file)

    sample=sample+1
  

  X_test.append(X_test_img)
  X_test.append(X_test_pose)
  

  return X_train, y_train, X_val, y_val, X_test, y_test




def get_partition_convNext(set=1):
  # 1. Indexes of the images that will correspond to train and test
  #trainfrs = [1:300]; 
	#testfrs = [301:589];
  p1=[]
  p2=[]
  for i in range(1, 590):
    if i in range(1,301):
      p1.append(i)
    if i in range(301,590):
      p2.append(i)
  
  # 2. Set 
  if set==1:
    train_val=p1
    test=p2
  else:
    train_val=p2
    test=p1

  # TRAIN/VALIDATION/TEST
  train=[]
  validation=[]

  for i in range(0, len(train_val)):
    if i in [1,11,21,31,41,51,61,71,81,91,101,111,121,131,141,151,161,171,181,191,201,211,221,231,241,251,261,271,281,291]:
      validation.append(train_val[i])
    else:
      train.append(train_val[i])


  return train, validation, test
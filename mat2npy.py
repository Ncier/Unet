from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import cv2
import scipy
import datetime
import scipy.io   as scio

import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from skimage import  img_as_float32
 

test_image=r"C:\Users\Administrator\Desktop\1000\预测图像"
test_mask =r"C:\Users\Administrator\Desktop\1000\预测图像"


image_training=r"F:\深度学习项目\zzq\本地CPU-绿潮代码\Data_zoo\MIT_SceneParsing\ADEChallengeData2016\images\training"#validation training
mask_training=r"F:\深度学习项目\zzq\本地CPU-绿潮代码\Data_zoo\MIT_SceneParsing\ADEChallengeData2016\annotations\training"#validation training
image_validation=r"F:\深度学习项目\zzq\本地CPU-绿潮代码\Data_zoo\MIT_SceneParsing\ADEChallengeData2016\images\validation"#validation training
mask_validation=r"F:\深度学习项目\zzq\本地CPU-绿潮代码\Data_zoo\MIT_SceneParsing\ADEChallengeData2016\annotations\validation"#validation training

#image_training=r"F:\深度学习项目\绿潮数据集\模板2\images\training"#validation training
#mask_training=r"F:\深度学习项目\绿潮数据集\模板2\annotations\training"#validation training
#image_validation=r"F:\深度学习项目\绿潮数据集\模板2\images\validation"#validation training
#mask_validation=r"F:\深度学习项目\绿潮数据集\模板2\annotations\validation"#validation training



##mat-mat
def geneTrainNpy(image_path,mask_path):
    image_name_training= glob.glob(os.path.join(image_path,"*.mat"))
    mask_name_training  = glob.glob(os.path.join(mask_path,"*.mat"))
#    image_name_validation= glob.glob(os.path.join(image_ptraining,"*.mat"))
#    mask_name_validation  = glob.glob(os.path.join(mask_training,"*.mat"))
    
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_training):
        #img = cv2.imread(item, -1)
        img = scio.loadmat(item)               
        image=(img['data'])/255
        
        #img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
        mask = scio.loadmat(mask_name_training[index])
        masks=(mask['data'])
        masks = img_as_float32(masks)
        #mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
        masks = masks[:, :,np.newaxis]
       
        image_arr.append(image)
        mask_arr.append(masks)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr

##mat-png
def geneTrainNpy2(test_image,test_mask):
    image_name_training= glob.glob(os.path.join(test_image,"*.mat"))
    mask_name_training  = glob.glob(os.path.join(test_mask,"*.png"))
#    image_name_validation= glob.glob(os.path.join(image_ptraining,"*.mat"))
#    mask_name_validation  = glob.glob(os.path.join(mask_training,"*.mat"))
    
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_training):
        #img = cv2.imread(item, -1)
        img = scio.loadmat(item)               
        image=(img['data'])/255
        
        
#        mask = scio.loadmat(mask_name_training[index])
#        masks=(mask['data'])
#        masks = img_as_float32(masks)       
#        masks = masks[:, :,np.newaxis]
       
        masks =  cv2.imdecode(np.fromfile(mask_name_training[index],dtype=np.uint8),cv2.IMREAD_LOAD_GDAL)
        masks = img_as_float32(masks)
        masks = masks[:, :,np.newaxis]   
                                      
        image_arr.append(image)
        mask_arr.append(masks)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr





##mat-mat
#image_trainings,mask_trainings=geneTrainNpy(image_training,mask_training)
#np.save(r"C:\Users\Administrator\Desktop\Unet2019-03-16\GreenDate\images_training.npy",image_trainings)
#np.save(r"C:\Users\Administrator\Desktop\Unet2019-03-16\GreenDate\annotations_training.npy",mask_trainings)
#
#
#image_validations,mask_validations=geneTrainNpy(image_validation,mask_validation)
#np.save(r"C:\Users\Administrator\Desktop\Unet2019-03-16\GreenDate\images_validation.npy",image_validations)
#np.save(r"C:\Users\Administrator\Desktop\Unet2019-03-16\GreenDate\annotations_validation.npy",mask_validations)


##mat-png
#image_trainings,mask_trainings=geneTrainNpy2(image_training,mask_training)
#np.save(r"C:\Users\Administrator\Desktop\Unet2019-03-16\datas\images_training.npy",image_trainings)
#np.save(r"C:\Users\Administrator\Desktop\Unet2019-03-16\datas\annotations_training.npy",mask_trainings)
#
#
#image_validations,mask_validations=geneTrainNpy2(image_validation,mask_validation)
#np.save(r"C:\Users\Administrator\Desktop\Unet2019-03-16\datas\images_validation.npy",image_validations)
#np.save(r"C:\Users\Administrator\Desktop\Unet2019-03-16\datas\annotations_validation.npy",mask_validations)


##test-mat-png
test_images,test_masks=geneTrainNpy2(test_image,test_mask)
np.save(r"C:\Users\Administrator\Desktop\Unet2019-03-16\test\test_images.npy",test_images)
np.save(r"C:\Users\Administrator\Desktop\Unet2019-03-16\test\test_masks.npy",test_masks)





























from mynet import *
from model import unet
from data import *
from Deeplabv3P import*
from utils import plotHistory, plotFeaturemap
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import os
import scipy.misc
from PIL import Image
import numpy as np
import sklearn.metrics as metrics
import glob
import sys
from keras import backend as K

from osgeo import gdal_array
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
data_dir = 'logs/unet_pse-2019-05-05_12-05'


def modelTest(modelPath, i, IMAGE_SIZE,total):
    '''
    load the weighted, predict the test results.
    '''
    LR = 0.001
    epochs = 1
    batch_size = 1
    # =============================================================================
    # 模型
    #model = myXceptFCN(2, (IMAGE_SIZE,IMAGE_SIZE,3), epochs, batch_size, LR, Falg_summary=False, Falg_plot_model=False)
    
    # model = myUnet(2, (IMAGE_SIZE,IMAGE_SIZE,3), epochs, batch_size, LR, Falg_summary=False, Falg_plot_model=False,
                   # pretrained_weights=modelPath)
    model = Deeplabv3(2, (IMAGE_SIZE,IMAGE_SIZE,2), epochs, batch_size, LR, Falg_summary=False, Falg_plot_model=False,
                   pretrained_weights=modelPath)
    # model = unet(2, (IMAGE_SIZE,IMAGE_SIZE,3), epochs, batch_size, LR, Falg_summary=False, Falg_plot_model=False,
                   # pretrained_weights=modelPath)
    # =============================================================================
    test_img = np.load('test/test_images.npy')
    test_Gt  = np.load('test/test_masks.npy')
    test_img = test_img[:, 0:IMAGE_SIZE, -IMAGE_SIZE:, :]
    test_Gt  = test_Gt[:, 0:IMAGE_SIZE, -IMAGE_SIZE:, :]
    
    # test_img = np.load('dataset/test/cuts/test_all_img.npy')
    # test_Gt  = np.load('dataset/test/cuts/test_all_Gt.npy')
    # =============================================================================
    # 预测
    print('%d / %d' % (i, total))
    
    scores = model.evaluate(test_img, test_Gt, verbose=0)
    print('Test loss:', scores[0])
    print('Test accu:', scores[1])

    y_predict = model.predict(test_img)
    K.clear_session()
    # =============================================================================
    # 保存预测图像
    modelName = modelPath.split('.h')[0:-1]
    savePath = modelName[0] + '-Accu-' + str(scores[1])[:6] + 'Pred.tif'
    
    
    
    saveImg(y_predict, savePath)
    #gdal_array.SaveArray(y_predict,savePath,format="GTIFF")  
    
    print('save predict image')
    
    # 绘制FeatureMap
    # if not os.path.isdir(modelName[0]):
        # os.makedirs(modelName[0])
    # plotFeaturemap(test_img, model, modelName[0])

def results(y, y_):
    precision = metrics.precision_score(y, y_)
    recall    = metrics.recall_score(y, y_)
    f1        = metrics.f1_score(y, y_)
    kappa     = metrics.cohen_kappa_score(y, y_)
    print('precision_score: %f'% (precision))
    print('recall_score: %f'% (recall))
    print('f1_score: %f'% (f1))
    print('kappa: %f'% (kappa))
    print(' ')

def print_f1(groundTruth, f_name, i, width, IMAGE_SIZE, total):
    print('%d / %d' % (i, total))

    pred_unet = np.array(Image.open(f_name)).astype('float32')
    pred_unet = pred_unet.reshape((width*width, 1))
    
    f_name_ = f_name.split('improvement-')[1]
    print(f_name_)
    
    # 规范0和255
    for i in range(width*width):
        if pred_unet[i] < 100:
            pred_unet[i] = 0
        else:
            pred_unet[i] = 255
            
    # 矩阵各个元素乘255，得到（0，1）矩阵
    #groundTruth   = groundTruth / 255
    pred_unet     = pred_unet / 255
    # 计算 混淆矩阵
    confusion_matr_unet   = metrics.confusion_matrix(groundTruth, pred_unet)
    # 计算 P,R,F1
    results(groundTruth, pred_unet)
    
class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass
    
f_names_hdf5 = glob.glob(data_dir + '/*.hdf5')
f_names_hdf5 = f_names_hdf5[0:]
for i, f_name in enumerate(f_names_hdf5):
    modelTest(f_name, i+1, IMAGE_SIZE = 600, total=len(f_names_hdf5))
    
# save log
Log_fileName = str(data_dir) + "/F1.log"
sys.stdout = Logger(Log_fileName)
print("<<-------------- START -------------->>")
# 读入图像，转换为矩阵
#groundTruth   = np.array(Image.open('./fcn128-2018-12-14/gt_1.png')).astype('float32')
groundTruth  = np.load('test/test_masks.npy')
# 图像大小，长宽相等
width = groundTruth.shape[1]
# 将矩阵转化为一列
groundTruth   = groundTruth.reshape((width*width, 1))

f_names_jpg = glob.glob(data_dir + '/*Pred.jpg')
f_names_jpg = f_names_jpg[0:]
for i, f_name in enumerate(f_names_jpg):
    print_f1(groundTruth, f_name, i+1, width, IMAGE_SIZE = 1024, total=len(f_names_jpg))
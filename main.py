from mynet import *
from model import unet
from data import *
from Deeplabv3P import*
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from utils import plotHistory, plotFeaturemap
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

IMAGE_SIZE = 24
LR = 0.0001
epochs = 50
batch_size = 2
# =============================================================================
# 模型
#model = unet(2, (IMAGE_SIZE,IMAGE_SIZE,3), epochs, batch_size, LR, Falg_summary=True, Falg_plot_model=False)
#model = myUnet(2, (IMAGE_SIZE,IMAGE_SIZE,3), epochs, batch_size, LR, Falg_summary=True, Falg_plot_model=False)
model = unet3(2, (IMAGE_SIZE,IMAGE_SIZE,2), epochs, batch_size, LR, Falg_summary=True, Falg_plot_model=False)
#model = myXceptFCN(2, (IMAGE_SIZE,IMAGE_SIZE,3), epochs, batch_size, LR, Falg_summary=True, Falg_plot_model=False)
# =============================================================================
savePath = mkSaveDir('unet_pse')
# 使用保存点
checkpointPath= savePath + "/unet-v1-improvement-{epoch:03d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(checkpointPath, monitor='val_acc', verbose=1,
                             save_best_only=False, save_weights_only=True, mode='auto', period=1)
EarlyStopping = EarlyStopping(monitor='val_acc', patience=50, verbose=1)
tensorboard = TensorBoard(log_dir=savePath, histogram_freq=0)
callback_lists = [tensorboard, EarlyStopping, checkpoint]
# model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
# =============================================================================
# True: 读取图像;False: Train with npy file
readImg = False
if readImg:
    myGene = trainGenerator(2,'dataset/train','images','groundTruth',save_to_dir = None)
    model.fit_generator(myGene,steps_per_epoch=30,epochs=epochs,callbacks=callback_lists)
else:
    train_image, train_GT, valid_image, valid_GT = readNpy()
    History = model.fit(train_image, train_GT, batch_size=batch_size, validation_data=(valid_image, valid_GT),
        epochs=epochs, verbose=1, shuffle=True, class_weight='auto', callbacks=callback_lists)
    with open(savePath + '/log_128.txt','w') as f:
        f.write(str(History.history))
#model.save_weights(savePath + '/save_weights.h5')

# 绘制accurate和loss曲线
plotHistory(History, savePath)
# 绘制FeatureMap
#plotFeaturemap(valid_image[0:1], model, savePath)
# =============================================================================
# 预测
# f_names = glob.glob(savePath + '/*.hdf5')
# for i, f_name in enumerate(f_names):
    # modelTest(f_name, IMAGE_SIZE = 1024)

# =============================================================================
from data import *

## create .npy data
#If your computer has enough memory, you can create npy files containing all your images and masks, and feed your DNN with them.
IMAGE_SIZE = 24

# train:
train_image_arr,train_mask_arr = geneTrainNpy("dataset/train/images/","dataset/train/groundTruth/", IMAGE_SIZE)
np.save("dataset/train/npy/train_image_arr.npy",train_image_arr)
np.save("dataset/train/npy/train_groundTruth_arr.npy",train_mask_arr)

# train:
# valid_image_arr,valid_mask_arr = geneTrainNpy("dataset/validation/validation-images/","dataset/validation/validation-groundTruth/", IMAGE_SIZE)
# np.save("dataset/train/npy/valid_image_arr.npy",valid_image_arr)
# np.save("dataset/train/npy/valid_groundTruth_arr.npy",valid_mask_arr)

# test:
# test_image_arr,test_mask_arr = geneTrainNpy("dataset/test/images/","dataset/test/groundTruth/", IMAGE_SIZE)
# np.save("dataset/train/npy/testImg_arr_1325.npy",test_image_arr)
# np.save("dataset/train/npy/testGt_arr_1325.npy",test_mask_arr)









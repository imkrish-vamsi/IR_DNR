# !python -m pip install 'fsspec>=0.3.3'
# !python -m pip install dask[bag] --upgrade
#!unzip "/content/drive/MyDrive/Colab Notebooks/SRGAN/RawData/CCPD2020.zip" -d "/content/drive/MyDrive/Colab Notebooks/SRGAN/Data/"
#!python "/content/drive/MyDrive/Colab Notebooks/SRGAN/Utilities/preprocess.py" 5 '/content/drive/MyDrive/Colab Notebooks/SRGAN/Data/CCPD2020/ccpd_green' '/content/drive/MyDrive/Colab Notebooks/SRGAN/PreprocessedData'
import sys
sys.path.append('X:\Upwork\projects\SRGAN\Utilities')

# Restart runtime if modules are not loading
from iofile import DataLoader
from lossMetric import *
from RRDBNet import RRDBNet
from GAN import Discriminator
from trainVal import MinMaxGame
import numpy as np
from glob import glob
from painter import Visualizer
import cv2
import os

# Model Training
PATH = 'X:\Upwork\projects\SRGAN\PreprocessedData\192_96' # only use images with shape 192 by 96 for training
files = glob.glob(PATH + '/*.jpg') * 3  # data augmentation, same image with different brightness and contrast
np.random.shuffle(files)
train, val = files[:int(len(files)*0.8)], files[int(len(files)*0.8):]
loader = DataLoader()
trainData = DataLoader().load(train, batchSize=16)
valData = DataLoader().load(val, batchSize=64)
discriminator = Discriminator()
extractor = buildExtractor()
generator = RRDBNet(blockNum=10)
# loss function as in the SRGAN paper
def contentLoss(y_true, y_pred):
    featurePred = extractor(y_pred)
    feature = extractor(y_true)
    mae = tf.reduce_mean(tfk.losses.mae(y_true, y_pred))
    return 0.1*tf.reduce_mean(tfk.losses.mse(featurePred, feature)) + mae

optimizer = tfk.optimizers.Adam(learning_rate=1e-3)
generator.compile(loss=contentLoss, optimizer=optimizer, metrics=[psnr, ssim])

# PSNR=20/ssim=0.65.
history = generator.fit(x=trainData, validation_data=valData, epochs=30, steps_per_epoch=300, validation_steps=100)

PARAMS = dict(lrGenerator = 1e-4, 
              lrDiscriminator = 1e-4,
              epochs = 1000, 
              stepsPerEpoch = 500, 
              valSteps = 100)
game = MinMaxGame(generator, discriminator, extractor)
log, valLog = game.train(trainData, valData, PARAMS)

# save training weights
generator.save_weights('X:\Upwork\projects\SRGAN\weights\rrdb', save_format='tf')

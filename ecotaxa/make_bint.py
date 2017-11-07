import skimage.io
import os
import tensorflow as tf
import numpy as np
from scipy.misc import imsave
import sys
import pylab
import matplotlib.image as mpimg

living_inputdir = 'cifar10_data/uvp5_ccelter_simple/1/'
not_living_inputdir = 'cifar10_data/uvp5_ccelter_simple/2/'
living_eval = 'cifar10_data/living_eval/'
not_living_eval = 'cifar10_data/not_living_eval/'
outdir = 'test_batch'

def write(image, outdir):
  newfile = open(outdir, 'ab')
  byte = bytes(image)
  print(sys.getsizeof(image))
  newfile.write(byte)
  newfile.close()

def make_train_bin(i):
  listoflive = os.listdir(living_inputdir)
  listofnotlive = os.listdir(not_living_inputdir)
  filenamel = living_inputdir  + listoflive[i]
  filenamen = not_living_inputdir  + listofnotlive[i]
  imagel = do_something(filenamel,i,0)
  imagen = do_something(filenamen,i,10)
  return imagel, imagen

def do_something(filename,p,y):
  image = skimage.io.imread(filename)
  h0 = image.shape[0]
  w0 = image.shape[1]
  npimage = image[:,:,1].reshape(h0,w0)
  temp = np.zeros([h0,w0]) 
  for i in range(h0):
    for j in range(w0):
      temp[i][j] = npimage[i][j]
  for i in range(w0):
    temp[0][i] = 0
    temp[h0-1][i] = 0
  for i in range(h0):
    temp[i][0] = 0
    temp[i][w0-1] = 0

  npimage = 255 - npimage
  if h0 < 44:
    temp = np.zeros([44,w0]) 
    for i in range(h0):
      for j in range(w0):
        temp[i][j] = npimage[i][j]
    npimage = temp
    h = 44
  else:
    h = h0

  if w0 < 44:
    temp = np.zeros([h,44]) 
    for i in range(h0):
      for j in range(w0):
        temp[i][j] = npimage[i][j]
    w = 44
    npimage = temp
  else:
    w = w0
  mass = 0
  weight = 0
  height = 0
  m11 = 0
  m20 = 0
  m02 = 0
  for i in range(h):
    for j in range(w):
      mass += npimage[i][j]
      height += npimage[i][j]*i

  for j in range(w):
    for i in range(h):
      weight += npimage[i][j]*j 

  height = int(height/mass)
  weight = int(weight/mass)

  if 0>(weight-23):
    left = 0
  else:
    if (weight+23)>w:
      left = w-45
    else:
      left = weight-23 

  if 0>(height-23):
    top = 0
  else:
    if (height+23)>h:
      top = h-45
    else:
      top = height-23 

  npimage = 255 - npimage
  temp = np.zeros([44,44],np.uint8)
  for k in range(44):
    for t in range(44):
      temp[k][t] = npimage[top +k][left+t]
  imsave('imagel%d.jpg'%(p+y),temp)
  return temp
"""
def draw():
  imagel = np.zeros([46,500])+255
  imagen = np.zeros([46,500])+255
  for i in range(10):
    t = mpimg.imread('imagel%d.jpg'%i)
    for j in range(46):
      imagel[:][i+j]= t[:][j]
  imsave('imagel.jpg',imagel)
  imsave('imagen.jpg',imagen)
"""
def main():
  imagel = np.zeros([10,46,46])
  imagen = np.zeros([10,46,46])
  k = [1,100,1000,10000,1555,4584,4561,14444,2000,8000]
  for i in range(10):
    imagel[i],imagen[i]=make_train_bin(k[i])
    #imsave('imagel%d.jpg'%i,imagel[i])
    #imsave('imagen%d.jpg'%i,imagen[i])

for i in range(5):
  make_train_bin(i)



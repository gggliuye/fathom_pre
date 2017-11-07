import skimage.io
import os
import tensorflow as tf
import numpy as np

living_inputdir = 'cifar10_data/uvp5_ccelter_simple/living/'
not_living_inputdir = 'cifar10_data/uvp5_ccelter_simple/not_living/'
living_eval = 'cifar10_data/living_eval/'
not_living_eval = 'cifar10_data/not_living_eval/'
outdir = 'ecotaxa_data_ex/train_batch'
outdireval = 'ecotaxa_data_ex/eval_batch'


def write(image, outdirx):
  newfile = open(outdirx, 'ab')
  byte = bytes(image)
  newfile.write(byte)
  newfile.close()


def make_train_bin():
  listoflive = os.listdir(living_inputdir)
  listofnotlive = os.listdir(not_living_inputdir)
  for i in range(len(listoflive)):
    filenamel = living_inputdir  + listoflive[i]
    filenamen = not_living_inputdir  + listofnotlive[i]
    imagel = do_something(filenamel)
    imagen = do_something(filenamen)
    #image size (45,44)
    x = np.zeros([1],np.uint8)+1
    if (listoflive[i][0] == '1') or (listoflive[i][0] == '2'):
      x[0] = (int(listoflive[i][1:3]) + 50)/25
    if (listoflive[i][0] == '3') or (listoflive[i][0] == '4'):
      x[0] = (int(listoflive[i][1:3]) + 150)/25
    write(x,outdir)
    write(imagel, outdir)

    x[0]=0
    write(x,outdir)
    write(imagen, outdir)
    if i%50 == 0:
      print("make %d bin file success"%i)

def make_eval_bin():
  listoflive = os.listdir(living_eval)
  listofnotlive = os.listdir(not_living_eval)
  for i in range(10000):
    filenamel = living_eval  + listoflive[i]
    filenamen = not_living_eval  + listofnotlive[i]
    imagel = do_something(filenamel)
    imagen = do_something(filenamen)
    #image size (45,44)
    x = np.zeros([1],np.uint8)+1
    if (listoflive[i][0] == '1') or (listoflive[i][0] == '2'):
      x[0] = (int(listoflive[i][1:3]) + 50)/25
    if (listoflive[i][0] == '3') or (listoflive[i][0] == '4'):
      x[0] = (int(listoflive[i][1:3]) + 150)/25
    write(x,outdireval)
    write(imagel, outdireval)
    x[0] = 0
    write(x,outdireval)
    write(imagen, outdireval)
    if i%50 == 0:
      print("make %d bin file success"%i)

def do_something(filename):
  image = skimage.io.imread(filename)
  h0 = image.shape[0]
  w0 = image.shape[1]
  npimage = image[:,:,1].reshape(h0,w0)
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

  temp = np.zeros([46,46],np.uint8)
  for i in range(44):
    for j in range(44):
      temp[i+1][j] = npimage[top+i][left+j]

  if mass >= 255000:
    x = 255
  else:  
    x = mass/2000
  
  for i in range(45):
    temp[i][44] = x
    temp[i][45] = x
    temp[0][i] = h0
    temp[45][i] = w0
    

  return temp


make_eval_bin()


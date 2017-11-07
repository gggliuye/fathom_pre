import skimage.io
import os
import tensorflow as tf
import numpy as np
import random

living_inputdir1 = 'cifar10_data/uvp5_ccelter_simple/1/'
living_inputdir2 = 'cifar10_data/uvp5_ccelter_simple/2/'
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
  listoflive1 = os.listdir(living_inputdir1)
  listoflive2 = os.listdir(living_inputdir2)
  listofnotlive = os.listdir(not_living_inputdir)
  for i in range(250):
    filenamel1 = living_inputdir1  + listoflive1[i]
    filenamel2 = living_inputdir2  + listoflive2[2*i]
    filenamel3 = living_inputdir2  + listoflive2[2*i+1]
    b = random.randrange(0,49999)
    filenamen = not_living_inputdir  + listofnotlive[b]
    filenamen0 = not_living_inputdir  + listofnotlive[b+1]
    imagel1 = do_something(filenamel1)
    imagel2 = do_something(filenamel2)
    imagel3 = do_something(filenamel3)
    imagen = do_something(filenamen)
    imagen0 = do_something(filenamen0)
    #image size (45,44)
    x = np.zeros([1],np.uint8)
    x[0]=1
    write(x,outdir)
    write(imagel1, outdir)

    x[0]=2
    write(x,outdir)
    write(imagel2, outdir)

    x[0]=2
    write(x,outdir)
    write(imagel3, outdir)

    x[0]=0
    write(x,outdir)
    write(imagen, outdir)

    x[0]=0
    write(x,outdir)
    write(imagen0, outdir)
    if i%30 == 0:
      print("make %d bin file success"%i)

def make_eval_bin():
  listoflive1 = os.listdir(living_inputdir1)
  listoflive2 = os.listdir(living_inputdir2)
  listofnotlive = os.listdir(not_living_inputdir)
  for k in range(50):
    i = k+250
    filenamel1 = living_inputdir1  + listoflive1[i]
    filenamel2 = living_inputdir2  + listoflive2[2*i]
    filenamel3 = living_inputdir2  + listoflive2[2*i+1]
    b = random.randrange(0,49999)
    filenamen = not_living_inputdir  + listofnotlive[b]
    filenamen0 = not_living_inputdir  + listofnotlive[b+1]
    imagel1 = do_something(filenamel1)
    imagel2 = do_something(filenamel2)
    imagel3 = do_something(filenamel3)
    imagen = do_something(filenamen)
    imagen0 = do_something(filenamen0)
    #image size (45,44)
    x = np.zeros([1],np.uint8)
    x[0]=1
    write(x,outdireval)
    write(imagel1, outdireval)

    x[0]=2
    write(x,outdireval)
    write(imagel2, outdireval)

    x[0]=2
    write(x,outdireval)
    write(imagel3, outdireval)

    x[0]=0
    write(x,outdireval)
    write(imagen, outdireval)

    x[0]=0
    write(x,outdireval)
    write(imagen0, outdireval)
    if i%30 == 0:
      print("make %d bin file success"%i)
"""
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

    x[0]=0
    write(x,outdireval)
    write(imagen, outdireval)
    if i%50 == 0:
      print("make %d bin file success"%i)
"""
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
  m11 = 0
  m20 = 0
  m02 = 0
  for i in range(h):
    for j in range(w):
      mass += npimage[i][j]
      height += npimage[i][j]*i
      m11 += npimage[i][j]*i*j
      m20 += npimage[i][j]*i*i
      m02 += npimage[i][j]*j*j

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
      temp[i+1][j+1] = npimage[top+i][left+j]

  if mass >= 25500:
    x = 255
  else:  
    x = mass/200
  
  t = np.array([[m20/mass,m11/mass],[m11/mass,m02/mass]])
  dig = np.diag(t)
  a = dig[0]/50
  b = dig[1]/50
  if a > 255: a =255
  if b>255: b=255
  for i in range(45):
    temp[i][0] = x
    temp[i][45] = x
    temp[0][i] = a
    temp[45][i] = b
    

  return temp


make_eval_bin()
make_train_bin()

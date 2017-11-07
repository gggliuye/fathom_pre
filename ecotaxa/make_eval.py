
import os
import random
import shutil

living_inputdir = 'cifar10_data/uvp5_ccelter_simple/living/'
nonliving_inputdir = 'cifar10_data/uvp5_ccelter_simple/not_living/'
living_train = 'cifar10_data/living_train'
living_eval = 'cifar10_data/living_eval'
not_living_train = 'cifar10_data/not_living_train'
not_living_eval = 'cifar10_data/not_living_eval'

def make_eval():
  listofliving = os.listdir(living_inputdir)
  listofnonliving = os.listdir(nonliving_inputdir)
  n = 0
  while n < 15966:
    i = random.randint(0,3)
    ind = 4 * n + i 
    living_file = living_inputdir +'/' + listofliving[ind]
    notliving_file = nonliving_inputdir +'/' + listofnonliving[ind]
    shutil.move(living_file, living_eval +'/' + listofliving[ind])
    shutil.move(notliving_file, not_living_eval +'/' + listofnonliving[ind])
    n += 1
    
make_eval()



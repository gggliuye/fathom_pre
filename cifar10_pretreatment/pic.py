import pickle
import pylab
import numpy as np
from scipy.misc import imsave
 

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dictory = pickle.load(fo, encoding='bytes')
    return dictory  


def showimage(ind):
  """
  fo = open('eval_data','rb')
  eval_data = pickle.load(fo)
  fo.close()
  """

  file = "F:/pre/cifar-10-batches-py/test_batch"
  dic = unpickle(file)
  data = dic[b'data']
  image = data[ind].reshape(32,32,3)
  print(image.shape)
  
  rgb = np.zeros((32, 32, 3), dtype=np.uint8)
  rgb[..., 0] = data[ind][0:1024].reshape(32,32)
  rgb[..., 1] = data[ind][1024:2048].reshape(32,32)
  rgb[..., 2] = data[ind][2048:3072].reshape(32,32)
  imsave('image_test.jpg', rgb)
  print(rgb.shape)


  #imsave('image_test.png', image)  
  """
  fig = pylab.figure()
  a1 = fig.add_subplot(231)
  a2 = fig.add_subplot(232)
  a3 = fig.add_subplot(233)
  a4 = fig.add_subplot(234)
  a5 = fig.add_subplot(235)
  a6 = fig.add_subplot(236)
  a1.imshow(data[ind][0:1024].reshape(32,32))
  a2.imshow(data[ind][1024:2048].reshape(32,32))
  a3.imshow(data[ind][2048:3072].reshape(32,32))
  a4.imshow(eval_data[ind][0:1024].reshape(32,32))
  a5.imshow(eval_data[ind][1024:2048].reshape(32,32))
  a6.imshow(eval_data[ind][2048:3072].reshape(32,32))
  pylab.axis("off")
  pylab.show()
  #pylab.savefig(figureFileName
  """
showimage(20)
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim.nets
import pickle
import pylab
from scipy.misc import imsave


slim = tf.contrib.slim


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
  label = dic[b'labels'][ind]
  image = data[ind].reshape(32,32,3)
  
  
  rgb = np.zeros((32, 32, 3), dtype=np.uint8)
  rgb[..., 0] = data[ind][0:1024].reshape(32,32)
  rgb[..., 1] = data[ind][1024:2048].reshape(32,32)
  rgb[..., 2] = data[ind][2048:3072].reshape(32,32)
  imsave('image_test.jpg', rgb)
  print(rgb.shape)

  return  label


def run(image_size, ind):
    
    label = showimage(ind)
    print(label)
    image_name = 'image_test.jpg'
    image_string = open(image_name, 'rb').read()
    image = tf.image.decode_jpeg(image_string, channels=3)

    if image.dtype != tf.float32:
       image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    
    height = image_size
    width = image_size

    distorted_image1 = tf.random_crop(image, [height, width,3])

    # Randomly flip the image horizontally.
    distorted_image2 = tf.image.random_flip_left_right(distorted_image1)
 
    # Because these operations are not commutative, consider randomizing
    # randomize the order their operation.
    distorted_image3 = tf.image.random_brightness(distorted_image2,  max_delta=63)
    
    distorted_image4 = tf.image.random_contrast(distorted_image3,lower=0.2, upper=1.8)
  
    #Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(distorted_image4)
    aa = tf.sigmoid(float_image)
    # Save the processed image for Fathom
    with tf.Session() as sess:
        i1 = image.eval()
        i2 = distorted_image1.eval()
        i3 = distorted_image2.eval()
        i4 = distorted_image3.eval()
        i5 = distorted_image4.eval()
        i6 = float_image.eval()
        i8 = aa.eval()
    i7 = i6
    for i in range(24):
      for j in range(24):
        for k in range(3):
          if i7[i][j][k] > 1:
            i7[i][j][k]=1
          if i7[i][j][k] < -1:
            i7[i][j][k]=-1    

    fig = pylab.figure()
    a1 = fig.add_subplot(131)
    a1.set_title("standardization")
    a2 = fig.add_subplot(132)
    a2.set_title("sigmoid")
    a3 = fig.add_subplot(133)
    a3.set_title("truncation")
    a1.imshow(i6)
    a2.imshow(i8)
    a3.imshow(i7)
    
    pylab.show()
    #fig.savefig('temp.png',dpi=fig.dpi)
    
run(24,24)

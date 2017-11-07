import numpy as np
import scipy.fftpack
import scipy.ndimage.filters

def truncateNonNeg(x):
    index = x.shape
    y = np.zeros([index[0],index[1]])
    for i in range(index[0]):
        for j in range(index[1]):
            if x[i][j] > 0:
                y[i][j] = x[i][j]
    return y


def getSamplePS(sample):
    samplePS = np.fft.fft2(sample)
    samplePS = scipy.fftpack.fftshift(samplePS)
    samplePS = (abs(samplePS)/(sample.shape[0]*sample.shape[1]))**2
    return samplePS
  
def getPowerSpectrumWhiteningFilter(averagePS,noiseVariance):
    M = averagePS.shape[0]*averagePS.shape[1] 
    temp = (averagePS - noiseVariance*M)/averagePS
    temp = truncateNonNeg(temp)
    temp /= np.sqrt(averagePS)
    temp = np.fft.ifftshift(temp)
    w = np.fft.ifft2(temp)
    w = np.real(w)
    w = np.fft.fftshift(w)
    return w


def whiteningtf(sample):
    for i in range(3):
      temp = sample[i*1024:i*1024+1024]
      image = temp.reshape((32,32))
      imagePS = getSamplePS(image)
      maxPS = np.max(imagePS)
      noiseVariance = maxPS*10**(-8) 
      filter = getPowerSpectrumWhiteningFilter(imagePS,noiseVariance)
      imageC = scipy.ndimage.filters.convolve(image, filter, mode='wrap')
      imageC = imageC.reshape((1,1024))
      if i == 0:
        imageCV = imageC
      else:
        imageCV = np.vstack((imageCV,imageC)) 
    return imageCV.reshape((1,3072))



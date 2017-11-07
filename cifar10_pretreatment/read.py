import whitening

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dictory = pickle.load(fo, encoding='bytes')
    return dictory  

def dataset(inputdict):
  import numpy
  for i in range(5):
    filename = "data_batch_%d" % (i+1)
    file = inputdict + filename
    dic = unpickle(file)
    if i == 0:
      a = dic[b'data']
    else:
      a = numpy.vstack((a,dic[b'data'])) 
  """
  k = 0
  for i in range(len(a)):
    a[i] = whitening.whiteningtf(a[i])
    if i/100 > k:
       print(k)
       k = k + 1
  """
  return a 

def labels(inputdict):
  for i in range(5):
    filename = "data_batch_%d" % (i+1)
    file = inputdict + filename
    dic = unpickle(file)
    if i == 0:
      b = dic[b'labels']
    else:
      b = b + dic[b'labels']
  return b 

def dataset_test(inputdict):
  filename = "test_batch"
  file = inputdict + filename
  dic = unpickle(file)
  a = dic[b'data']
 # for i in range(len(a)):
 #   a[i] = whitening.whiteningtf(a[i])
  return a


def labels_test(inputdict):
  filename = "test_batch"
  file = inputdict + filename
  dic = unpickle(file)
  return dic[b'labels']


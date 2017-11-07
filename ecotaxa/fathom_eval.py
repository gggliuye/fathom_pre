import numpy as np
import os
import os.path
import tensorflow as tf
import tensorflow.contrib.slim.nets
import cifar10 as cifar10
import skimage.io
from scipy.misc import imsave

slim = tf.contrib.slim

def run(name):
    
    image_name = 'living.jpg'
    image_string = open(image_name, 'rb').read()
    image = tf.image.decode_jpeg(image_string, channels=1)

    if image.dtype != tf.float32:
       image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    
    image = tf.expand_dims(image, 0)
    
    # Save the processed image for Fathom
    with tf.Session() as sess:
        npimage = image.eval()
    np.save('image.npy', npimage.squeeze())

    # Run the network with Tensorflow
    with tf.Graph().as_default():
        image = tf.placeholder("float", [1,45,44,1], name="input")

        #with slim.arg_scope(inception.inception_v1_arg_scope()):
        logits = cifar10.inference(image)
        #probabilities = tf.nn.softmax(logits)
        probabilities = logits
        #init_fn = slim.assign_from_checkpoint_fn('./model.ckpt-19999', slim.get_model_variables('model'))
        
        variable_averages = tf.train.ExponentialMovingAverage(cifar10.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        with tf.Session() as sess:
          ckpt = tf.train.get_checkpoint_state('ecotaxa_train/')
          saver.restore(sess, ckpt.model_checkpoint_path)
        init_fn = slim.assign_from_checkpoint_fn('./ecotaxa_train/model.ckpt-0', tf.global_variables())

        with tf.Session() as sess:
            init_fn(sess)
            prob = probabilities.eval(feed_dict = {"input:0" : npimage})
            # Save the network for Fathom
            saver = tf.train.Saver(tf.global_variables())
            saver.save(sess, name)
        """
        i = 1
        for n in tf.get_default_graph().as_graph_def().node:
          if i < 100:
             print("name:",n.name)
             print("op",n.op)
             print("input:",n.input)
             print("    ")
          i = i +1
        """
    # Dump output
    prob = prob.squeeze()
    probidx = np.argsort(-prob)
    print('Top-2:')
    for i in range(2):
        print(probidx[i], prob[probidx[i]])
    # Run Fathom on same network and same processed image
     

   # python3 ./Fathom_p35/Fathom.cpython-35.pyc validate --network-description examples/test_debug/inference.meta --output-validation-type top-5 --output-node Softmax --output-expected-id 3 --image examples/test_debug/image.npy

    
run('inference')

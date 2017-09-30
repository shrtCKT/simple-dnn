import sys
import os.path
# sys.path.insert(0, os.path.abspath("./simple-dnn"))

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import scipy.misc
import time

class GeneratorDC(object):
    def __init__(self, x_dims, x_ch, g_conv_units, 
                 g_kernel_sizes=[5,5], g_strides=[2, 2], g_paddings='SAME',
                 g_activation_fn=tf.nn.relu):
      """
      DCGAN Generator network. 
        :param x_dims:  2d list [width, height]; the x dimentions.
        :param x_ch: int; the channels in x.
        :param g_conv_units: a list; the number of channels in each conv layer.
        :param g_kernel_sizes: A list of length 2 [kernel_height, kernel_width], for all the conv layer filters.
                               Or a list of list, each list of size if size of the filter per cov layer. 
        :param g_strides: a list of tuples, each tuple holds the number stride of each conv layer.
                          or 2d list in which case all the conv layers will have the same stride.
        :param g_paddings: string or list of strings, specifying the padding type.
        :param g_activation_fn: a single or a list of activations functions.
      """
    
    
      # Data Config
      self.x_dims = x_dims
      self.x_ch = x_ch

      ######################## Generator
      self.g_conv_units = g_conv_units
      
      if isinstance(g_kernel_sizes[0], list) or isinstance(g_kernel_sizes[0], tuple):
          assert len(g_conv_units) == len(g_kernel_sizes)
          self.g_kernel_sizes = g_kernel_sizes
      else:
          self.g_kernel_sizes = [g_kernel_sizes] * len(g_conv_units)
      
      if isinstance(g_strides[0], list) or isinstance(g_strides[0], tuple):
          assert len(g_conv_units) == len(g_strides)
          self.g_strides = g_strides
      else:
          self.g_strides = [g_strides] * len(g_conv_units)
          
      if isinstance(g_paddings, list):
          assert len(g_conv_units) == len(g_paddings)
          self.g_paddings = g_paddings
      else:
          self.g_paddings = [g_paddings] * len(g_conv_units)
          
      self.g_activation_fn = g_activation_fn


    def __call__(self, z, ys):
        z_concat = tf.concat([z, ys], axis=1)
        zP = slim.fully_connected(
            z_concat, 4*4*256, normalizer_fn=slim.batch_norm,
            activation_fn=tf.nn.relu,scope='g_project', 
            weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
        zCon = tf.reshape(zP,[-1,4,4,256])
        net = zCon
        
        with slim.arg_scope([slim.conv2d_transpose], 
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                            normalizer_fn=slim.batch_norm,
                            activation_fn=self.g_activation_fn):
            for i, (g_unit, kernel_size, stride, padding) in enumerate(zip(
                    self.g_conv_units, self.g_kernel_sizes, self.g_strides, self.g_paddings)):
                net = slim.conv2d_transpose(net, num_outputs=g_unit, kernel_size=kernel_size,
                                            stride=stride, padding=padding, scope='g_conv{0}'.format(i))
        
        g_out = slim.convolution2d_transpose(
            net,num_outputs=self.x_ch, kernel_size=self.x_dims, padding="SAME",
            biases_initializer=None,activation_fn=tf.nn.tanh,
            scope='g_out', weights_initializer=tf.truncated_normal_initializer(stddev=0.02))

        return g_out

class DiscriminatorDC(object):
  def __init__(self, y_dim,
               conv_units, 
               hidden_units=None,
               kernel_sizes=[5,5], strides=[2, 2], paddings='SAME',
               d_activation_fn=tf.contrib.keras.layers.LeakyReLU,     # Conv Layers
               f_activation_fns=tf.nn.relu,                           # Fully connected
               dropout=False, keep_prob=0.5):
    self.y_dim = y_dim
    ######################## Discremenator
    # Conv layer config
    self.conv_units = conv_units
    if isinstance(kernel_sizes[0], list) or isinstance(kernel_sizes[0], tuple):
      assert len(conv_units) == len(kernel_sizes)
      self.kernel_sizes = kernel_sizes
    else:
      self.kernel_sizes = [kernel_sizes] * len(conv_units)
    
    if isinstance(strides[0], list) or isinstance(strides[0], tuple):
      assert len(conv_units) == len(strides)
      self.strides = strides
    else:
      self.strides = [strides] * len(conv_units)
      
    if isinstance(paddings, list):
      assert len(conv_units) == len(paddings)
      self.paddings = paddings
    else:
      self.paddings = [paddings] * len(conv_units)
      
    self.d_activation_fn = d_activation_fn
    
    # Fully connected layer config
    self.hidden_units = hidden_units
    if not isinstance(f_activation_fns, list) and  self.hidden_units is not None:
      self.f_activation_fns = [f_activation_fns] * len(self.hidden_units)
    else:
      self.f_activation_fns = f_activation_fns

    ######################## Training Config
    self.dropout = dropout
    self.keep_prob = keep_prob
    self.matching_layer = None
      
  def build_conv(self, x, reuse=False):
      net = x
      with slim.arg_scope([slim.conv2d], padding='SAME',
                          weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                          activation_fn=self.d_activation_fn(alpha=0.2),
                            # weights_regularizer=slim.l2_regularizer(0.0005),
                          reuse=reuse):
        fm_layer = None
        for i, (c_unit, kernel_size, stride, padding) in enumerate(zip(
            self.conv_units, self.kernel_sizes, self.strides, self.paddings)):
          # Conv
          net = slim.conv2d(net, c_unit, kernel_size, stride=stride, normalizer_fn=slim.batch_norm,
                                  padding=padding, scope='d_conv{0}'.format(i))
          if self.matching_layer is not None and i == self.matching_layer:
              fm_layer = net
              
          # Dropout: Do NOT use dropout for conv layers. Experiments show it gives poor result.
      return net, fm_layer
  
  def build_full(self, net, reuse=False):
      with slim.arg_scope([slim.fully_connected], 
                          weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
        for i, (h_unit, activation_fn) in enumerate(zip(
            self.hidden_units, self.f_activation_fns)):
          net = slim.fully_connected(net, h_unit, normalizer_fn=slim.batch_norm,
                                     activation_fn=activation_fn,
                                     reuse=reuse, scope='d_full{0}'.format(i))
          if self.dropout:
            net = slim.dropout(net, keep_prob=self.keep_prob, is_training=self.is_training)
      return net
      
  def __call__(self, x, reuse=False, logits=True, matching_layer=None):
    self.matching_layer = matching_layer
    # Conv Layers
    net, fm_layer = self.build_conv(x, reuse=reuse)
    # Flatten Conv Layer Output
    net = slim.flatten(net)

    # Fully Connected Layers
    if self.hidden_units is not None:
        net = self.build_full(net, reuse=reuse)

    # Output logits
    if logits:
        d_out = slim.fully_connected(net, self.y_dim + 1, activation_fn=None,
                                      reuse=reuse, scope='d_out', 
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
    else:
        d_out = slim.fully_connected(net, self.y_dim + 1, activation_fn=tf.nn.sigmoid,
                                      reuse=reuse, scope='d_out', 
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
    return d_out, fm_layer


class ConditionalGAN(object):
    """ Implementation of Deep Convolutional Conditional Generative Adversarial Network.
    """
    def __init__(self, 
                 x_dims, x_ch, y_dim,
                 generator=None,     # Generator Net
                 discriminator=None, # Discriminator Net
                 x_reshape=None,
                 x_scale=None,
                 x_inverse_scale=None,
                 z_dim=100,
                 d_optimizer=tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5),
                 g_optimizer=tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5),
                 g_loss_fn='default',
                 g_target=1.0,
                 d_label_smooth=0.75,
                 sigmoid_alpha=10,
                 l2_penalty=0.01,
                 batch_size=128, iterations=2000,
                 display_step=100, save_step=1000,
                 oracle=None,
                 graph=None, sess=None,
                 sample_directory=None, #Directory to save sample images from generator in.
                 model_directory=None, #Directory to save trained model to.
                ):
        """
        Args:
        x_dims - list; the width of and hight of image x.
        x_ch - int; the number of channels(depth) of input x.
        y_dim - int; number of data labeles.        
        """
        # Data Config
        self.x_dims = x_dims
        self.x_ch = x_ch
        self.y_dim = y_dim
        self.z_size = z_dim
        self.x_reshape = x_reshape
        if x_scale is not None or x_inverse_scale is not None:
          # If one is not none the both should be not none 
          assert x_scale is not None and x_inverse_scale is not None 
        
        self.x_scale = x_scale
        self.x_inverse_scale = x_inverse_scale
        
        ######################## Generator and Discriminator Networks
        self.generator = generator
        self.discriminator = discriminator

        ######################## Training config
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        assert g_loss_fn in ['default', 'smoothed', 'sigmoid', 'feature_matching', 
                             'feature_default', 'l2_default', 'least_square']
        self.g_loss_fn = g_loss_fn
        if self.g_loss_fn == 'feature_matching' or self.g_loss_fn == 'feature_default':
            assert matching_layer == -1 or matching_layer < len(conv_units)
            self.matching_layer = matching_layer if matching_layer != -1 else len(conv_units) - 1
        self.sigmoid_alpha = sigmoid_alpha
        self.d_label_smooth = d_label_smooth
        self.g_target = g_target
        self.iterations = iterations
        self.batch_size = batch_size
        self.display_step = display_step
        self.save_step = save_step
        self.sample_directory = sample_directory
        self.model_directory = model_directory
        self.oracle = oracle
        self.l2_penalty = l2_penalty
        
        if graph:
            self.graph = graph
        else:
            self.graph = tf.Graph()
            
        with self.graph.as_default():
            self.build_model()
            
            if sess:
                self.sess = sess
            else:
                self.sess = tf.Session()
                
            # To save and restore checkpoints.
            self.saver = tf.train.Saver()
    
    def x_reformat(self, xs):
      """ Rescale and reshape x if x_scale and x_reshape functions are provided.
      """
      if self.x_scale is not None:
        xs = self.x_scale(xs)
      if self.x_reshape is not None:
        xs = self.x_reshape(xs)
      return xs

    @staticmethod
    def sigmoid_cost(input, alpha):
        exp = tf.exp(-alpha * (input - 0.5))
        return tf.divide(1.0, 1 + exp)
    
    def build_model(self):
      with self.graph.as_default():
        # Placeholders
        self.z_in = tf.placeholder(shape=[None,self.z_size], dtype=tf.float32) #Random vector
        self.real_in = tf.placeholder(
          shape=[None, self.x_dims[0], self.x_dims[1], 1], dtype=tf.float32) #Real images
        self.real_label = tf.placeholder(
          shape=[None, self.y_dim + 1], dtype=tf.float32) #real image labels
        self.fake_label = tf.placeholder(
          shape=[None, self.y_dim + 1], dtype=tf.float32) #fake image labels
        # One side D label smoothing
        self.real_label = self.real_label * self.d_label_smooth

        self.Gz = self.generator(self.z_in, self.real_label) # Condition generator on real labels
        self.Dx, fm_layer_x = self.discriminator(
          self.real_in, logits=True, 
          matching_layer=self.matching_layer if self.g_loss_fn == 'feature_matching' else None)
        self.Dg, fm_layer_g = self.discriminator(
          self.Gz, reuse=True, logits=True,
          matching_layer=self.matching_layer if self.g_loss_fn == 'feature_matching' else None)

        Dx_softmax = tf.nn.softmax(logits=self.Dx)
        Dg_softmax = tf.nn.softmax(logits=self.Dg)
        
        # d_loss and g_loss together define the optimization objective of the GAN.
        d_loss_real = tf.nn.softmax_cross_entropy_with_logits(logits=self.Dx, 
                                                              labels=self.real_label)
        d_loss_fake = tf.nn.softmax_cross_entropy_with_logits(logits=self.Dg, 
                                                              labels=self.fake_label)
        if self.g_loss_fn == 'least_square':
            ls_dx = 0.5 * tf.reduce_mean(tf.square(tf.subtract(Dx_softmax, self.real_label)))
            ls_dg = 0.5 * tf.reduce_mean(tf.square(tf.subtract(Dg_softmax, self.fake_label)))
            self.d_loss = ls_dx + ls_dg
        else:
            self.d_loss = tf.reduce_mean(d_loss_real + d_loss_fake)

        tvars = tf.trainable_variables()
        d_vars = [var for var in tvars if 'd_' in var.name]
        g_vars = [var for var in tvars if 'g_' in var.name]
                    
        if self.g_loss_fn == 'smoothed':
          self.g_loss = -tf.reduce_mean(
              (1 - self.g_target) * tf.log(Dg_softmax[:, -1]) + 
              self.g_target * tf.log(1. - Dg_softmax[:, -1])
          )
        elif self.g_loss_fn == 'sigmoid':
          self.g_loss = -tf.reduce_mean(tf.log(1 - ConditionalGAN.sigmoid_cost(
            Dg_softmax[:, -1], self.sigmoid_alpha)))
        elif self.g_loss_fn == 'feature_matching':
          self.g_loss = tf.reduce_mean(tf.square(tf.subtract(fm_layer_x, fm_layer_g)))
        elif self.g_loss_fn == 'feature_default':
          self.g_loss = -tf.reduce_mean(tf.log(1. - Dg_softmax[:, -1])) + \
                        tf.reduce_mean(tf.square(tf.subtract(fm_layer_x, fm_layer_g)))
        elif self.g_loss_fn == 'l2_default':
          g_l2_loss = 0.
          for w in g_vars:
              g_l2_loss += (self.l2_penalty * tf.reduce_mean(tf.nn.l2_loss(w)))
          self.g_loss = -tf.reduce_mean(tf.log(1. - Dg_softmax[:, -1])) + g_l2_loss
        elif self.g_loss_fn == 'least_square': # based on https://arxiv.org/abs/1611.04076
          self.g_loss = 0.5 * tf.reduce_mean(tf.square((1. - Dg_softmax[:, -1]) - 1))
        else:
          self.g_loss = -tf.reduce_mean(tf.log(1. - Dg_softmax[:, -1]))
                    
        # Compute gradients
        trainerD = self.d_optimizer
        trainerG = self.g_optimizer
        d_grads = trainerD.compute_gradients(self.d_loss, d_vars) #Only update the weights for the discriminator network.
        g_grads = trainerG.compute_gradients(self.g_loss, g_vars) #Only update the weights for the generator network.

        ## For Debuging
        d_grads_decomposed, _ = list(zip(*d_grads))
        g_grads_decomposed, _ = list(zip(*g_grads))
        self.d_grad_norm = tf.global_norm(d_grads_decomposed)
        self.g_grad_norm = tf.global_norm(g_grads_decomposed)
        self.d_w_norm = tf.global_norm(d_vars)
        self.g_w_norm = tf.global_norm(g_vars)
        ##
    
        self.update_D = trainerD.apply_gradients(d_grads)
        self.update_G = trainerG.apply_gradients(g_grads)
    
    def _save_samples(self, i):
        n_samples = 36
        ys = np.zeros([n_samples, self.y_dim + 1])
        ys[:, np.random.randint(0, self.y_dim, size=n_samples)] = 1
        generated_x = self.generate(ys)
        
        generate_imgs = ConditionalGAN.merge_img(
          np.reshape(generated_x[0:n_samples],[n_samples, self.x_dims[0], self.x_dims[1]]),
          [6,6])
        
        if not os.path.exists(self.sample_directory):
            os.makedirs(self.sample_directory)
        scipy.misc.imsave(self.sample_directory+'/fig'+str(i)+'.png', generate_imgs)
        
        
    def _next_batch(self, x, y):
        start_index = np.random.randint(0, x.shape[0] - self.batch_size)
        return x[start_index:(start_index + self.batch_size)], \
               y[start_index:(start_index + self.batch_size)]

    def _accuracy(self, val_x, val_y, reformat=True):
        pred_y = self.predict(val_x, reformat=reformat)
        return (np.argmax(val_y, axis=1) == pred_y).mean()
    
    def _iter_stats(self, i, start_time, gLoss, dLoss, 
                    xs=None, ys=None, zs=None, ys_fake=None,
                    val_x=None, val_y=None):
      d_grad_norm, g_grad_norm, oracle_x, d_w_norm, g_w_norm = self.sess.run(
        (self.d_grad_norm, self.g_grad_norm, self.Gz, self.d_w_norm, self.g_w_norm),
        feed_dict={self.z_in:zs, self.real_in:xs, 
                   self.real_label:ys, self.fake_label:ys_fake})
      
      tr_acc = None
      if xs is not None and ys is not None and ys_fake is not None:
        tr_x = np.concatenate((xs, oracle_x), axis=0)
        tr_y = np.concatenate((ys, ys_fake), axis=0)
        tr_acc = self._accuracy(tr_x, tr_y, reformat=False) 
      
      v_acc = None
      if val_x is not None and val_y is not None:
          v_acc = self._accuracy(val_x, val_y) 
      
      oracle_acc = None
      if self.oracle is not None:
          oracle_acc = self.oracle(oracle_x)
          
      if i == 0:
          print '{0:5}| {1:6}| {2:5}| {3:4}| {4:6}| {5:6}| {6:6}| {7:5}| {8:4}| {9:6}| {10:6}'.format(
              'i', 'GLOSS', 'DLOSS', 'TIME', 'GGRAD', 'DGRAD', 'TR_ACC','V_ACC', 'ORA', 'DW', 'GW')
      
      print '{0:5}| {1:5.3}| {2:5.3}| {3:4}s| {4}| {5}| {6}| {7}| {8}| {9}| {10}'.format(
          i, gLoss, dLoss, int(time.time()-start_time), 
          '      ' if g_grad_norm is None else '{:6.4}'.format(g_grad_norm),
          '      ' if d_grad_norm is None else '{:6.4}'.format(d_grad_norm),
          '      ' if tr_acc is None else '{:6.3}'.format(tr_acc),
          '     ' if v_acc is None else '{:5.3}'.format(v_acc),
          '    ' if oracle_acc is None else '{:4.2}'.format(oracle_acc),
          '      ' if d_w_norm is None else '{:6.4}'.format(d_w_norm),
          '      ' if g_w_norm is None else '{:6.4}'.format(g_w_norm))
  
    def fit(self, X, y=None, val_x=None, val_y=None):
        start = time.time()
        self.discriminator.is_training = True
        with self.graph.as_default():  
            self.sess.run(tf.global_variables_initializer())
            for i in range(self.iterations):
                zs = np.random.uniform(-1.0, 1.0,
                                       size=[self.batch_size, self.z_size]).astype(np.float32)
                xs, ys = self._next_batch(X, y)
                xs = self.x_reformat(xs)

                # Create space for the fake class label for the real data labels
                ys = np.concatenate((ys, np.zeros_like(ys[:,0])[:,None]), axis=1)
                # Create the labels for the generated data.
                ys_fake = np.zeros_like(ys)
                ys_fake[:,-1] = 1

                _,dLoss = self.sess.run(
                    [self.update_D, self.d_loss],
                    feed_dict={self.z_in:zs, self.real_in:xs, 
                               self.real_label:ys, self.fake_label:ys_fake})
                _,gLoss = self.sess.run(
                    [self.update_G, self.g_loss],
                    feed_dict={self.z_in:zs, self.real_in:xs,
                               self.real_label:ys, self.fake_label:ys_fake})
                if i % self.display_step == 0:
                    self._iter_stats(i, start, gLoss, dLoss, 
                                     xs=xs, ys=ys, zs=zs, ys_fake=ys_fake,
                                     val_x=val_x, val_y=val_y)
                    self._save_samples(i)
                if i % self.save_step == 0 and i != 0 and self.model_directory is not None:
                  self.save_model('model-'+str(i)+'.cptk')
                  print "Saved Model"
    
            self._iter_stats(i, start, gLoss, dLoss, 
                              xs=xs, ys=ys, zs=zs, ys_fake=ys_fake,
                              val_x=val_x, val_y=val_y)
            self._save_samples(i)
            if self.model_directory is not None:
              self.save_model('model-'+str(i)+'.cptk')
              print "Saved Model"
        self.discriminator.is_training = False
    
    def generate(self, ys):
      """ Generate samples.
      
        :param ys: one hot encoded class labels. 
      """ 
      n_samples = len(ys)
      self.discriminator.is_training = False
      zs = np.random.uniform(-1.0,1.0,
                              size=[n_samples,self.z_size]).astype(np.float32)
      generated_x = self.sess.run(self.Gz,
                                  feed_dict={self.z_in:zs, self.real_label:ys})
      self.discriminator.is_training = True
      return self.x_inverse_scale(generated_x) if self.x_inverse_scale is not None \
                                               else generated_x
    
    @staticmethod
    def merge_img(images, grid_size=[6,6]):
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * grid_size[0], w * grid_size[1]))

        for idx, image in enumerate(images):
            i = idx % grid_size[1]
            j = idx // grid_size[1]
            img[j*h:j*h+h, i*w:i*w+w] = image

        return img

    def predict(self, X, reformat=True):
        probs = self.predict_prob(X, reformat=reformat)
        return np.argmax(probs, axis=1)

    def predict_prob(self, X, reformat=True):
        self.discriminator.is_training = False
        with self.graph.as_default():
            if reformat:
                xs = self.x_reformat(X)
            else:
                xs = X
            probs = self.sess.run(tf.nn.softmax(logits=self.Dx), feed_dict={self.real_in:xs})
        self.discriminator.is_training = True
        return probs
    
    def save_model(self, model_file_name):
      if self.model_directory is None:
        return 'ERROR: Model directory is None'
      if not os.path.exists(self.model_directory):
          os.makedirs(self.model_directory)
      return self.saver.save(self.sess, os.path.join(self.model_directory, model_file_name))
    
    def restore_model(self, model_file):
        with self.graph.as_default():
            self.saver.restore(self.sess, model_file)



# ---------------------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------------------
from ..util.format import UnitPosNegScale, reshape_pad
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../../data/MNIST_data/", 
                                  one_hot=True)

print mnist.train.images.shape
print mnist.train.labels.shape

discriminator = DiscriminatorDC(10,  # y_dim
                              [16,32,64], # conv_units
                              hidden_units=None,
                              kernel_sizes=[5,5], strides=[2, 2], paddings='SAME',
                              d_activation_fn=tf.contrib.keras.layers.LeakyReLU,
                              f_activation_fns=tf.nn.relu,
                              dropout=False, keep_prob=0.5)
generator = GeneratorDC([32, 32],#x_dims
                        1, # x_ch
                        [64,32,16], # g_conv_units
                        g_kernel_sizes=[5,5], g_strides=[2, 2], g_paddings='SAME',
                        g_activation_fn=tf.nn.relu)



dcgan = ConditionalGAN([32, 32], # x_dim 
              1, # x_ch 
              10, # y_dim 
              z_dim=100,
              generator=generator,     # Generator Net
              discriminator=discriminator, # Discriminator Net
              x_reshape=reshape_pad([28,28], [32,32], 1, pad=True, pad_value=-1),
              x_scale=UnitPosNegScale.scale,
              x_inverse_scale=UnitPosNegScale.inverse_scale,
              d_optimizer=tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5),
              g_optimizer=tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5),
              g_loss_fn='default',
              d_label_smooth=0.75,
              ## Training config
              batch_size=128,
              iterations=5,
              display_step=1,
              save_step=500,
              sample_directory='../../data/figs/closed',
#               model_directory='../../data/models/closed'
             )

dcgan.fit(mnist.train.images, mnist.train.labels,
          val_x=mnist.validation.images, val_y=mnist.validation.labels)

n_samples = 36
ys_gen = np.zeros([n_samples, mnist.train.labels.shape[1] + 1])
ys_gen[:, np.random.randint(0, mnist.train.labels.shape[1], size=n_samples)] = 1
        
gen_xs = dcgan.generate(ys_gen)
gen_imgs = ConditionalGAN.merge_img(np.reshape(gen_xs[0:n_samples],[n_samples, dcgan.x_dims[0], dcgan.x_dims[1]]))
plt.imshow(gen_imgs, cmap='gray')
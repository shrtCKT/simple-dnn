""" The generator network of a GAN.
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim


class GeneratorDC(object):
    """ Deep Convolutional Generator.
    """
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


    def __call__(self, z, ys=None):
        if ys is None:
            z_concat = z
        else:
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

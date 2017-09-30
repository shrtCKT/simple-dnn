import numpy as np

def reshape_pad(input_shape_2d, desired_shape_2d, input_ch, pad=True, pad_value=-1):
  """
  Reshape a flattend 2d input matrix X into 4D. Where:
    * axis 0 is batch size.
    * axis 1 is image width.
    * axis 2 is image height.
    * axis 3 is number of image color channels.

    :param input_shape_2d: [input_image_width, input_image_height]
    :param desired_shape_2d: [desired_image_width, desired_image_height]
    :param input_ch:  input image channels.
    :param rescale=True: rescale input range between -1 and 1. 
                          Assuming the input range is already between 0 and 1.
    :param pad=True: pad the width and height with -1 to reformat to the desired shape.
  """
  assert (desired_shape_2d[0] - input_shape_2d[0]) % 2 == 0
  assert (desired_shape_2d[1] - input_shape_2d[1]) % 2 == 0
  axis_1_pad_size = (desired_shape_2d[0] - input_shape_2d[0]) / 2
  axis_2_pad_size = (desired_shape_2d[1] - input_shape_2d[1]) / 2
  
  def reshape(xs):
    """ Reshapes and paddes xs.
      :param xs: a 2d array
    """
    assert len(xs.shape) == 2
    batch_size = xs.shape[0]
    
    xs = np.reshape(xs, [batch_size, input_shape_2d[0], input_shape_2d[1], input_ch])
    if pad:
        xs = np.lib.pad(xs,((0,0), (axis_1_pad_size, axis_1_pad_size), 
                            (axis_2_pad_size, axis_2_pad_size), (0,0)), 
                        'constant', constant_values=(pad_value, pad_value))
    
    return xs   

  return reshape

class UnitPosNegScale(object):
  @staticmethod
  def scale(xs):
    """ Assumes xs is already in unit scale with range of [0, 1]
    """
    return (xs - 0.5) * 2.0 #Transform range between -1 and 1

  @staticmethod
  def inverse_scale(xs):
    return (xs + 1.) / 2.

def unit_scale(xs):
  """
  Scale each column in the range of [0, 1]
    :param xs: Input matrix.
  """
  assert len(xs.shape) == 2 
  xs_min = np.amin(xs, axis=0)
  xs_range = np.amax(xs, axis=0) - xs_min
  xs = np.true_divide(np.subtract(xs, xs_min), xs_range)
  xs[np.isnan(xs)] = 0
  return xs  

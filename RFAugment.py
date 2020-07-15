"""Image Augmentations to be applied to reals and fakes during GAN training, for both GPUs and TPUs.
   Batch functions from Zhao et al 2020's DiffAugment https://github.com/mit-han-lab/data-efficient-gans

   apply to real and fake images in loss function like so:

   reals = RFAugment.augment(reals, policy='zoom in', channels_first=True, mode='tpu')

   TODO: More augs here https://github.com/google/automl/blob/master/efficientdet/aug/autoaugment.py"""

import tensorflow as tf
import os
import math
import functools


def augment(batch, policy='', channels_first=True, mode='gpu', probability=None):
    batch_size = os.environ.get('BATCH_PER', '?')
    if probability is not None:
        probability = float(max(min(probability, 1), 0)) # clamp p to a float between 0 and 1
    else:
      batch_size = int(batch_size)
    batch.set_shape((batch_size, None, None, None))
    print('In shape:')
    print(batch.shape)
    if mode == 'gpu':
        print('Augmenting reals and fake in gpu mode')
        if policy:
            if channels_first:
                print('Transposing channels')
                batch = tf.transpose(batch, [0, 2, 3, 1])
            for pol in policy.split(','):
                pol = pol.replace(" ", "")
                if pol in BATCH_AUGMENT_FNS:
                    for f in BATCH_AUGMENT_FNS[pol]:
                        print('POLICY : ', pol)
                        if probability is None:
                            batch = f(batch)
                        if probability is not None:
                            random_apply(f, probability, batch)
                elif pol in SINGLE_IMG_FNS:
                    for f in SINGLE_IMG_FNS[pol]:
                        print('POLICY : ', pol)
                        if probability is None:
                            batch = tf.map_fn(f, batch)
                        else:
                            batch = tf.map_fn(lambda inp: random_apply(inp[0], inp[1], inp[2]), (f, probability, batch))
            if channels_first:
                batch = tf.transpose(batch, [0, 3, 1, 2])
        return batch
    elif mode == 'tpu':
        if batch_size == '?':
            print('BATCH SIZE: ', batch_size)
            raise Exception('Some augmentations need a static batch size to run.\n Please set environment variable BATCH_PER to your batch size')
        print('Augmenting reals and fake in tpu mode')
        if probability is not None:
            print('ERROR: Random applications may not work on tpu')
        if policy:
            if channels_first:
                print('Transposing channels')
                batch = tf.transpose(batch, [0, 2, 3, 1])
            for pol in policy.split(','):
                pol = pol.replace(" ", "")
                if pol in BATCH_AUGMENT_FNS_TPU:
                    for f in BATCH_AUGMENT_FNS_TPU[pol]:
                        print('POLICY : ', pol)
                        batch = f(batch)
                elif pol in SINGLE_IMG_FNS_TPU:
                    for f in SINGLE_IMG_FNS_TPU[pol]:
                        print('POLICY : ', pol)
                        batch = tf.map_fn(f, batch)
            if channels_first:
                batch = tf.transpose(batch, [0, 3, 1, 2])
        return batch


alpha_default = 0.1  # set default alpha for spatial augmentations
colour_alpha_default = 0.1  # set default alpha for colour augmentations
alpha_override = float(os.environ.get('SPATIAL_AUGS_ALPHA', '0'))
colour_alpha_override = float(os.environ.get('COLOUR_AUGS_ALPHA', '0'))
if alpha_override > 0:
    if alpha_override >= 1:
        alpha_override = 0.999
    print(f'Overriding default alpha setting - setting to {alpha_override}')
    alpha_default = alpha_override
if colour_alpha_override > 0:
    if colour_alpha_override >= 1:
        colour_alpha_override = 0.999
    print(f'Overriding default colour alpha setting - setting to {colour_alpha_override}')
    colour_alpha_default = colour_alpha_override

def op_scope(fn, name=None):
    if name is None:
        name = fn.__name__
    @functools.wraps(fn)
    def _fn(*args, **kwargs):
        with tf.name_scope(fn.__name__):
            return fn(*args, **kwargs)
    return _fn

# ----------------------------------------------------------------------------
# Util functions:


# TODO: why doesn't this work on tpu
def random_apply(func, p, x):
    """Randomly apply function func to x with probability p."""
    return tf.cond(
      tf.less(tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32),
              tf.cast(p, tf.float32)),
      lambda: func(x),
      lambda: x)


def _pad_to_bounding_box(img, offset_height, offset_width, target_height,
                        target_width):
    """Pad `image` with zeros to the specified `height` and `width`.
    Adds `offset_height` rows of zeros on top, `offset_width` columns of
    zeros on the left, and then pads the image on the bottom and right
    with zeros until it has dimensions `target_height`, `target_width`.
    This op does nothing if `offset_*` is zero and the image already has size
    `target_height` by `target_width`. [Doesn't work on tpu.]

    Args:
    image: 3-D Tensor of shape `[height, width, channels]`
    offset_height: Number of rows of zeros to add on top.
    offset_width: Number of columns of zeros to add on the left.
    target_height: Height of output image.
    target_width: Width of output image.
    Returns:
    3-D float Tensor of shape
    `[target_height, target_width, channels]`
    """
    shape = tf.shape(img)

    height = shape[0]
    width = shape[1]
    after_padding_width = target_width - offset_width - width
    after_padding_height = target_height - offset_height - height
    # Do not pad on the depth dimension.
    paddings = tf.reshape(tf.stack([offset_height, after_padding_height, offset_width, after_padding_width, 0, 0]), [3, 2])
    return tf.pad(img, paddings)


def round_up_to_even(f):
    return math.ceil(f / 2.) * 2


def get_matrix(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    """
    returns 3x3 transform matrix which transforms indices in the image tensor
    """

    # CONVERT DEGREES TO RADIANS
    rotation = math.pi * rotation / 180.
    shear = math.pi * shear / 180.

    # ROTATION MATRIX
    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')
    rotation_matrix = tf.reshape(tf.concat([c1, s1, zero, -s1, c1, zero, zero, zero, one], axis=0), [3, 3])

    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    shear_matrix = tf.reshape(tf.concat([one, s2, zero, zero, c2, zero, zero, zero, one], axis=0), [3, 3])

    # ZOOM MATRIX
    zoom_matrix = tf.reshape(
        tf.concat([one / height_zoom, zero, zero, zero, one / width_zoom, zero, zero, zero, one], axis=0), [3, 3])

    # SHIFT MATRIX
    shift_matrix = tf.reshape(tf.concat([one, zero, height_shift, zero, one, width_shift, zero, zero, one], axis=0),
                              [3, 3])

    return tf.tensordot(tf.tensordot(rotation_matrix, shear_matrix, 1), tf.tensordot(zoom_matrix, shift_matrix, 1), 1)

# ----------------------------------------------------------------------------
# GPU Batch augmentations:


def rand_brightness(x, alpha=colour_alpha_default):
    """
    apply random brightness to image or batch of images.

    :param x: 4-D batch tensor or 3-D tensor with a single image.
    :param alpha: float, strength of augmentation.
    :return: 4-D batch tensor or 3-D tensor with a single image.
    """
    magnitude = (tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) - 0.5) * alpha
    x = x + magnitude
    return x


def rand_color(x, alpha=colour_alpha_default):
    """
    apply random colour to image or batch of images.

    :param x: 4-D batch tensor or 3-D tensor with a single image.
    :param alpha: float, strength of augmentation.
    :return: 4-D batch tensor or 3-D tensor with a single image.
    """
    magnitude = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) * 2 * alpha
    x_mean = tf.reduce_mean(x, axis=3, keepdims=True)
    x = (x - x_mean) * magnitude + x_mean
    return x


def rand_contrast(x, alpha=colour_alpha_default):
    """
    apply random contrast to image or batch of images.

    :param x: 4-D batch tensor or 3-D tensor with a single image.
    :param alpha: float, strength of augmentation.
    :return: 4-D batch tensor or 3-D tensor with a single image.
    """
    magnitude = (tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) + 0.5) * alpha
    x_mean = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
    x = (x - x_mean) * magnitude + x_mean
    return x


def batch_cutout(x, alpha=alpha_default):
    """
    Vectorized / batch solution to random cutout adapted from DiffAugment

    :param x: 4-D batch tensor
    :param alpha: float, strength of augmentation.
    :param ratio:
    :return: 4-D batch tensor
    """
    batch_size = tf.shape(x)[0]
    image_size = tf.shape(x)[1:3]
    cutout_size = image_size * alpha
    offset_x = tf.random.uniform([tf.shape(x)[0], 1, 1], maxval=image_size[0] + (1 - cutout_size[0] % 2), dtype=tf.int32)
    offset_y = tf.random.uniform([tf.shape(x)[0], 1, 1], maxval=image_size[1] + (1 - cutout_size[1] % 2), dtype=tf.int32)
    grid_batch, grid_x, grid_y = tf.meshgrid(tf.range(batch_size, dtype=tf.int32), tf.range(cutout_size[0], dtype=tf.int32), tf.range(cutout_size[1], dtype=tf.int32), indexing='ij')
    cutout_grid = tf.stack([grid_batch, grid_x + offset_x - cutout_size[0] // 2, grid_y + offset_y - cutout_size[1] // 2], axis=-1)
    mask_shape = tf.stack([batch_size, image_size[0], image_size[1]])
    cutout_grid = tf.maximum(cutout_grid, 0)
    cutout_grid = tf.minimum(cutout_grid, tf.reshape(mask_shape - 1, [1, 1, 1, 3]))
    mask = tf.maximum(1 - tf.scatter_nd(cutout_grid, tf.ones([batch_size, cutout_size[0], cutout_size[1]], dtype=tf.float32), mask_shape), 0)
    x = x * tf.expand_dims(mask, axis=3)
    return x


def cutmix(image, alpha=alpha_default):
    """
    Randomly mixes images within a minibatch. Uses matrix multiplication to mix images,
    as variable shape tensors may not work on tpus.

    :param image:
    :param alpha: float, strength of augmentation.
    :return:
    """

    dimensions = image.shape[1]
    batch_size = image.shape[0]
    imgs = []

    foo = [image[i] for i in range(batch_size)]

    for j in range(batch_size):
        one = foo[j]
        i = tf.random.uniform([], minval=0, maxval=batch_size, dtype=tf.int32)
        two = tf.switch_case(i, {index: lambda: foo[index] for index in range(batch_size)})

        def do_cutout(one):
            mask = random_cutout_tpu(tf.ones_like(two), alpha=alpha)
            foreground = tf.multiply(two, 1 - mask)
            background = tf.multiply(one, mask)
            return tf.add(foreground, background)

        # don't do anything if randomly selected image is current image
        out = tf.cond(tf.reduce_all(tf.equal(j, i)), lambda: one, lambda: do_cutout(one))
        imgs.append(out)

    out = tf.reshape(tf.stack(imgs), (batch_size, dimensions, dimensions, 3))
    return out

BATCH_AUGMENT_FNS = {
    'color': [rand_brightness, rand_color, rand_contrast],
    'colour': [rand_brightness, rand_color, rand_contrast],  # American spelling is a crime
    'brightness': [rand_brightness],
    'batchcutout': [batch_cutout],
    'mirrorh': [tf.image.random_flip_left_right],
    'mirrorv': [tf.image.random_flip_up_down],
    'cutmix': [cutmix]
}

BATCH_AUGMENT_FNS_TPU = {
    'color': [rand_brightness, rand_color, rand_contrast],
    'colour': [rand_brightness, rand_color, rand_contrast],
    'mirrorh': [tf.image.random_flip_left_right],
    'mirrorv': [tf.image.random_flip_up_down],
    'cutmix': [cutmix]
}

# ----------------------------------------------------------------------------
# GPU only single image augmentations


def rand_crop(img, crop_h, crop_w, seed=None):
    """
    Custom implementation of tf.image.random_crop() without all the assertions.

    :param img: 3-D tensor with a single image.
    :param crop_h:
    :param crop_w:
    :param seed: seed for random functions
    :return:
    """
    shape = tf.shape(img)
    h, w = shape[0], shape[1]
    begin = [h - crop_h, w - crop_w] * tf.random.uniform([2], 0, 1, seed=seed)
    begin = tf.cast(begin, tf.int32)
    begin = tf.concat([begin, [0]], axis=0)  # Add channel dimension.
    image = tf.slice(img, begin, [crop_h, crop_w, 3])
    return image


def zoom_in(img, alpha=alpha_default, target_image_shape=None, seed=None):
    """
    Random zoom in to TF image
    Args:
      img: 3-D tensor with a single image.
      alpha: strength of augmentation
      target_image_shape: List/Tuple with target image shape.
      seed: seed for random functions, optional.
    Returns:
      Image tensor with shape `target_image_shape`.
    """
    if alpha_override > 0:
      alpha = alpha_override
    n = tf.random_uniform(shape=[], minval=1 - alpha, maxval=1, dtype=tf.float32, seed=seed, name=None)
    shape = tf.shape(img)
    h = shape[0]
    w = shape[1]
    h_t = tf.cast(
        h, dtype=tf.float32, name='height')
    w_t = tf.cast(
        w, dtype=tf.float32, name='width')
    rnd_h = h_t * n
    rnd_w = w_t * n
    if target_image_shape is None:
        target_image_shape = (h, w)

    # Random crop
    rnd_h = tf.cast(
        rnd_h, dtype=tf.int32, name='height')
    rnd_w = tf.cast(
        rnd_w, dtype=tf.int32, name='width')
    cropped_img = rand_crop(img, rnd_h, rnd_w, seed=seed)

    # resize back to original size
    resized_img = tf.image.resize(
        cropped_img, target_image_shape, method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False,
        name=None
    )

    return resized_img


def zoom_out(img, alpha=alpha_default, target_image_shape=None, seed=None):
    """
    Random zoom out of TF image
    Args:
      img: 3-D tensor with a single image.
      alpha: strength of augmentation
      target_image_shape: List/Tuple with target image shape.
      seed: seed for random functions, optional.
    Returns:
      Image tensor with shape `target_image_shape`.
    """
    if alpha_override > 0:
      alpha = alpha_override
    # Set params
    n = tf.random_uniform(shape=[], minval=0, maxval=alpha, dtype=tf.float32, seed=seed, name=None)

    shape = tf.shape(img)
    h = shape[0]
    w = shape[1]

    if target_image_shape is None:
        target_image_shape = (h, w)

    # Pad image to size (1+2a)*H, (1+2a)*W
    h_t = tf.cast(
        h, dtype=tf.float32, name=None)
    w_t = tf.cast(
        w, dtype=tf.float32, name=None)
    rnd_h = h_t * n
    rnd_w = w_t * n
    paddings = [[rnd_h, rnd_h], [rnd_w, rnd_w], [0, 0]]
    padded_img = tf.pad(img, paddings, 'REFLECT')

    # Random crop to size (1+a)*H, (1+a)*W
    rnd_h = (1 + n) * h_t
    rnd_w = (1 + n) * w_t
    rnd_h = tf.cast(
        rnd_h, dtype=tf.int32, name='height')
    rnd_w = tf.cast(
        rnd_w, dtype=tf.int32, name='width')
    cropped_img = rand_crop(padded_img, rnd_h, rnd_w, seed=seed)

    # Resize back to original size
    resized_img = tf.image.resize(
        cropped_img, target_image_shape, method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False,
        name=None
    )

    return resized_img


def x_translate(img, alpha=alpha_default, target_image_shape=None, seed=None):
    """
    Random X translation within TF image with reflection padding
    Args:
      img: 3-D tensor with a single image.
      alpha: strength of augmentation
      target_image_shape: List/Tuple with target image shape.
      seed: seed for random functions, optional.
    Returns:
      Image tensor with shape `target_image_shape`.
    """
    if alpha_override > 0:
      alpha = alpha_override
    n = tf.random_uniform(shape=[], minval=0, maxval=alpha, dtype=tf.float32, seed=seed, name=None)

    shape = tf.shape(img)
    h = shape[0]
    w = shape[1]

    if target_image_shape is None:
        target_image_shape = (h, w)

    # Pad image to size H, (1+2a)*W
    w_t = tf.cast(
        w, dtype=tf.float32, name=None)
    rnd_w = w_t * n
    paddings = [[0, 0], [rnd_w, rnd_w], [0, 0]]
    padded_img = tf.pad(img, paddings, 'REFLECT')

    # Random crop section at original size
    X_trans = rand_crop(padded_img, target_image_shape[0], target_image_shape[1], seed=seed)
    return X_trans


def xy_translate(img, alpha=alpha_default, target_image_shape=None, seed=None):
    """
    Random XY translation within TF image with reflection padding
    Args:
      img: 3-D tensor with a single image.
      alpha: strength of augmentation
      target_image_shape: List/Tuple with target image shape.
    Returns:
      Image tensor with shape `target_image_shape`.
    """
    if alpha_override > 0:
      alpha = alpha_override
    n = tf.random_uniform(shape=[], minval=0, maxval=alpha, dtype=tf.float32, seed=seed, name=None)
    shape = tf.shape(img)
    h = shape[0]
    w = shape[1]
    if target_image_shape is None:
        target_image_shape = (h, w)

    # Pad image to size (1+2a)*H, (1+2a)*W
    h_t = tf.cast(
        h, dtype=tf.float32, name=None)
    w_t = tf.cast(
        w, dtype=tf.float32, name=None)
    rnd_h = h_t * n
    rnd_w = w_t * n
    paddings = [[rnd_h, rnd_h], [rnd_w, rnd_w], [0, 0]]
    padded_img = tf.pad(img, paddings, 'REFLECT')

    # Random crop section at original size
    xy_trans = rand_crop(padded_img, target_image_shape[0], target_image_shape[1], seed=seed)
    return xy_trans


def y_translate(img, alpha=alpha_default, target_image_shape=None, seed=None):
    """
    Random Y translation within TF image with reflection padding
    Args:
      img: 3-D tensor with a single image.
      alpha: strength of augmentation
      target_image_shape: List/Tuple with target image shape.
    Returns:
      Image tensor with shape `target_image_shape`.
    """
    if alpha_override > 0:
      alpha = alpha_override
    n = tf.random_uniform(shape=[], minval=0, maxval=alpha, dtype=tf.float32, seed=seed, name=None)
    shape = tf.shape(img)
    h = shape[0]
    w = shape[1]

    if target_image_shape is None:
        target_image_shape = (h, w)

    # Pad image to size (1+2a)*H, W
    h_t = tf.cast(
        h, dtype=tf.float32, name=None)
    rnd_h = h_t * n
    paddings = [[rnd_h, rnd_h], [0, 0], [0, 0]]
    padded_img = tf.pad(img, paddings, 'REFLECT')

    # Random crop section at original size
    Y_trans = rand_crop(padded_img, target_image_shape[0], target_image_shape[1], seed=seed)
    return Y_trans


def random_cutout(img, alpha=alpha_default, seed=None):
    """
    Cuts random black square out of TF image
    Args:
    image: 3-D tensor with a single image.
    alpha: affects max size of square
    target_image_shape: List/Tuple with target image shape.
    Returns:
    Cutout Image tensor
    """
    if alpha_override > 0:
      alpha = alpha_override

    # get img shape
    shape = tf.shape(img)
    h = shape[0]
    w = shape[1]

    # get square of random shape less than w*a, h*a
    val = tf.cast(tf.minimum(h, w), dtype=tf.float32)
    max_val = tf.cast((alpha*val), dtype=tf.int32)
    size = tf.random_uniform(shape=[], minval=1, maxval=max_val, dtype=tf.int32, seed=seed, name=None)

    # get random xy location of square
    x_loc_upper_bound = w - size
    y_loc_upper_bound = h - size

    x = tf.random_uniform(shape=[], minval=0, maxval=x_loc_upper_bound, dtype=tf.int32, seed=seed, name=None)
    y = tf.random_uniform(shape=[], minval=0, maxval=y_loc_upper_bound, dtype=tf.int32, seed=seed, name=None)

    erase_area = tf.ones([size, size, 3], dtype=tf.float32)

    if erase_area.shape == (0, 0, 3):
        return img
    else:
        mask = 1.0 - _pad_to_bounding_box(erase_area, y, x, h, w)
        erased_img = tf.multiply(img, mask)
        return erased_img


def apply_random_zoom(x, seed=None):
    with tf.name_scope('RandomZoom'):
        x.set_shape(x.shape)
        choice = tf.random_uniform([], 0, 2, tf.int32, seed=seed)
        x = tf.cond(tf.reduce_all(tf.equal(choice, tf.constant(0))), lambda: zoom_in(x, seed=seed), lambda: zoom_out(x, seed=seed))
        return x

def apply_random_aug(x, seed=None):
    with tf.name_scope('RandomAugmentations'):
        x.set_shape(x.shape)
        choice = tf.random_uniform([], 0, 6, tf.int32, seed=seed)
        #TODO: change to less than to stop all the branching
        x = tf.cond(tf.reduce_all(tf.equal(choice, tf.constant(0))), lambda: zoom_in(x, seed=seed), lambda: x)
        x = tf.cond(tf.reduce_all(tf.equal(choice, tf.constant(1))), lambda: zoom_out(x, seed=seed), lambda: x)
        x = tf.cond(tf.reduce_all(tf.equal(choice, tf.constant(2))), lambda: x_translate(x, seed=seed), lambda: x)
        x = tf.cond(tf.reduce_all(tf.equal(choice, tf.constant(3))), lambda: y_translate(x, seed=seed), lambda: x)
        x = tf.cond(tf.reduce_all(tf.equal(choice, tf.constant(4))), lambda: xy_translate(x, seed=seed), lambda: x)
        x = tf.cond(tf.reduce_all(tf.equal(choice, tf.constant(5))), lambda: random_cutout(x, seed=seed), lambda: x)
        return x


SINGLE_IMG_FNS = {
    'zoomin': [zoom_in],
    'zoomout': [zoom_out],
    'randomzoom': [apply_random_zoom],
    'xtrans': [x_translate],
    'ytrans': [y_translate],
    'xytrans': [xy_translate],
    'cutout': [random_cutout],
    'random': [apply_random_aug]
}

# ----------------------------------------------------------------------------
# TPU & GPU compatible single image augmentations.


def zoom_in_tpu(img, alpha=alpha_default):
    x_offset = tf.random.uniform([], minval=-alpha, maxval=alpha)
    y_offset = tf.random.uniform([], minval=-alpha, maxval=alpha)
    zoomlvl = tf.random.uniform([], minval=1-alpha, maxval=alpha)
    return tf_image_translate(img, x_offset, y_offset, zoom=zoomlvl, data_format="NHWC", wrap_mode="reflect")


def zoom_out_tpu(img, alpha=alpha_default):
    x_offset = tf.random.uniform([], minval=-alpha, maxval=alpha)
    y_offset = tf.random.uniform([], minval=-alpha, maxval=alpha)
    zoomlvl = tf.random.uniform([], minval=1, maxval=1+alpha)
    return tf_image_translate(img, x_offset, y_offset, zoom=zoomlvl, data_format="NHWC", wrap_mode="reflect")


def random_zoom_tpu(img, alpha=alpha_default):
    x_offset = tf.random.uniform([], minval=-alpha, maxval=alpha)
    y_offset = tf.random.uniform([], minval=-alpha, maxval=alpha)
    zoomlvl = tf.random.uniform([], minval=1-alpha, maxval=1+alpha)
    return tf_image_translate(img, x_offset, y_offset, zoom=zoomlvl, data_format="NHWC", wrap_mode="reflect")


def x_translate_tpu(img, alpha=alpha_default):
    x_offset = tf.random.uniform([], minval=-alpha, maxval=alpha)
    y_offset = 0
    return tf_image_translate(img, x_offset, y_offset, zoom=1.0, data_format="NHWC", wrap_mode="reflect")


def y_translate_tpu(img, alpha=alpha_default):
    x_offset = 0
    y_offset = tf.random.uniform([], minval=-alpha, maxval=alpha)
    return tf_image_translate(img, x_offset, y_offset, zoom=1.0, data_format="NHWC", wrap_mode="reflect")


def xy_translate_tpu(img, alpha=alpha_default):
    x_offset = tf.random.uniform([], minval=-alpha, maxval=alpha)
    y_offset = tf.random.uniform([], minval=-alpha, maxval=alpha)
    return tf_image_translate(img, x_offset, y_offset, zoom=1.0, data_format="NHWC", wrap_mode="reflect")


def random_rotate(image):
    return matrix_transform(image, zoomopt='none', transopt='', rotate=True)


def random_shear(image):
    return matrix_transform(image, zoomopt='none', transopt='', shear=True)


def random_cutout_tpu(img, alpha=alpha_default):
    """
    Random cutout applied using matrix transformations.

    :param img:
    :param alpha:
    :return:
    """
    # make blank img of same size
    assert img.shape[0] % 2 == 0
    assert img.shape[1] % 2 == 0
    dimensions = int(os.environ.get('RESOLUTION', '256'))
    x = round_up_to_even(dimensions * alpha)
    y = round_up_to_even(dimensions * alpha)
    blank = tf.ones([x, y, 3])
    padshapeX = tf.cast((dimensions - x) / 2, dtype=tf.int32)
    padshapeY = tf.cast((dimensions - x) / 2, dtype=tf.int32)
    paddings = [[padshapeY, padshapeY], [padshapeX, padshapeX], [0, 0]]
    padded_blank = tf.pad(blank, paddings, constant_values=0)
    mask = 1 - matrix_transform(padded_blank, alpha=1 - alpha, zoomopt='zoomout', transopt='xy', rotate=True, shear=True)
    erased_img = tf.multiply(img, mask)
    return erased_img


def matrix_transform(image, alpha=alpha_default, zoomopt='zoomin', transopt='xy', rotate=False, shear=False):
    """
    Does various matrix based transformations of a 3-D image tensor.
    Pads edges with the values at that edge in the original image.

    :param image: 3-D tensor with a single image.
    :param alpha: Strength of augmentation.
    :param zoomopt: Options for zoom. 'zoomin', 'zoomout', or 'none'.
    :param transopt: Options for translation. 'x', 'y', or 'xy'.
    :param rotate: if True, randomly rotate image.
    :param shear: if True, randomly shear image.
    :return: 3-D tensor with a single image.
    """
    dimensions = int(os.environ.get('RESOLUTION', '256'))
    XDIM = dimensions % 2  # fix for size 331

    if rotate:
        rot = 15. * tf.random.normal([1], dtype='float32') * alpha
    else:
        rot = tf.constant([0], dtype='float32')

    if shear:
        shr = 5. * tf.random.normal([1], dtype='float32') * alpha
    else:
        shr = tf.constant([0], dtype='float32')

    if zoomopt == 'zoomin':
        h_zoom = 1. + tf.random.uniform([1], dtype='float32') * alpha
        w_zoom = h_zoom
    elif zoomopt == 'zoomout':
        h_zoom = 1. - tf.random.uniform([1], dtype='float32') * alpha
        w_zoom = h_zoom
    elif zoomopt == 'both':
        h_zoom = 1. - tf.random.uniform([1], minval=1-alpha, maxval=1+alpha, dtype='float32') * alpha
        w_zoom = h_zoom
    else:
        h_zoom = tf.constant([1.])
        w_zoom = h_zoom

    if 'y' in transopt:
        # TODO: not sure these values are right
        h_shift = (tf.random.uniform([1], dtype='float32') - 0.5) * dimensions * alpha
    else:
        h_shift = tf.constant([0.])

    if 'x' in transopt:
        w_shift = (tf.random.uniform([1], dtype='float32') - 0.5) * dimensions * alpha
    else:
        w_shift = tf.constant([0.])

    # GET TRANSFORMATION MATRIX
    m = get_matrix(rot, shr, h_zoom, w_zoom, h_shift, w_shift)

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat(tf.range(dimensions // 2, -dimensions // 2, -1), dimensions)
    y = tf.tile(tf.range(-dimensions // 2, dimensions // 2), [dimensions])
    z = tf.ones([dimensions * dimensions], dtype='int32')
    idx = tf.stack([x, y, z])

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = tf.tensordot(m, tf.cast(idx, dtype='float32'), 1)
    idx2 = tf.cast(idx2, dtype='int32')
    idx2 = tf.clip_by_value(idx2, -dimensions // 2 + XDIM + 1, dimensions // 2)

    # FIND ORIGIN PIXEL VALUES
    idx3 = tf.stack([dimensions // 2 - idx2[0,], dimensions // 2 - 1 + idx2[1,]])
    d = tf.gather_nd(image, tf.transpose(idx3))

    return tf.reshape(d, [dimensions, dimensions, 3])


# Below fns are all untested / experimental

def sprinkles(img, alpha=alpha_default):
    """
    Implementation of https://medium.com/@lessw/progressive-sprinkles-a-new-data-augmentation-for-cnns-and-helps-achieve-new-98-nih-malaria-6056965f671a
    basically random_cutout multiple times, smaller
    :param img:
    :param alpha:
    :return:
    """
    # TODO: this can be much more efficient
    a = alpha / 5
    for i in range(5):
        img = random_cutout_tpu(img, alpha=a)
    return img


def apply_random_aug_tpu(x):
    """
    Apply random spatial augmentation to TF image.
    Options: zoom in, zoom out, X/Y/XY translate, random cutout.
    :param x: 3-D Tensor of shape `[height, width, channels]`
    :return: 3-D Tensor of shape `[height, width, channels]`
    """
    # TODO: why does this work, but random_apply doesn't?
    x.set_shape(x.shape)
    choice = tf.random_uniform([], 0, 3, tf.int32)

    def rz(): return random_zoom_tpu(x)

    def xy(): return xy_translate_tpu(x)

    def rc(): return random_cutout_tpu(x)

    #from Key: {i: (lambda x : lambda : func(x))(N) for i,func in enumerate([a,b])} ?

    return tf.switch_case(choice, branch_fns={0: rz, 1: xy, 2: rc})


def p_cutout(x):
    return random_apply_tpu(x, random_cutout, 0.50)


def random_apply_tpu(x, f, p):
    choice = tf.random_uniform([], 0, 3, tf.int32)

    def do_fn(): return f(x)

    def nothing(): return x

    fns = [do_fn] * int(p*10)
    while len(fns) < 100:
        fns.append(nothing)
    return tf.switch_case(choice, branch_fns={i: x for i, x in enumerate(fns)})


SINGLE_IMG_FNS_TPU = {
    'zoomin': [zoom_in_tpu],
    'zoomout': [zoom_out_tpu],
    'randomzoom': [random_zoom_tpu],
    'xtrans': [x_translate_tpu],
    'ytrans': [y_translate_tpu],
    'xytrans': [xy_translate_tpu],
    'cutout': [random_cutout_tpu],
    'random': [apply_random_aug_tpu],
}

# TODO: implement gridmask from here: https://www.kaggle.com/xiejialun/gridmask-data-augmentation-with-tensorflow

# ----------------------------------------------------------------------------
# TF image rasterizer by Shawwn: https://github.com/shawwn/tfimg/blob/master/aug.py

@op_scope
def ti32(x):
  return tf.cast( x, tf.int32 )


@op_scope
def tf32(x):
  return tf.cast( x, tf.float32 )


@op_scope
def clamp(v, min=0., max=1.):
  return tf.clip_by_value(v, min, max)


@op_scope
def wrap(v, wrap_mode="reflect"):
  assert wrap_mode in ["clamp", "wrap", "reflect"]
  if wrap_mode == "wrap":
    return tf.math.floormod(v, 1.0)
  elif wrap_mode == "reflect":
    return 1.0 - tf.abs(tf.math.floormod(v, 2.0) - 1.0)
  elif wrap_mode == "clamp":
    return clamp(v)


@op_scope
def iround(u):
  return ti32(tf.math.floordiv(tf32(u), 1.0))


@op_scope
def tf_image_translate(img, x_offset, y_offset, zoom=1.0, data_format="NHWC", wrap_mode="reflect"):
  # "NCHW" not implemented for now; handled by transpose in tf_image_translate
  #assert data_format in ["NHWC", "NCHW"]
  assert data_format in ["NHWC"]
  shape = tf.shape(img)
  if data_format == "NHWC":
    h, w, c = shape[0], shape[1], shape[2]
  else:
    c, h, w = shape[0], shape[1], shape[2]

  DUDX = 1.0 / tf32(w) * zoom
  DUDY = 0.0
  DVDX = 0.0
  DVDY = 1.0 / tf32(h) * zoom

  X1 = 0
  Y1 = 0
  U1 = x_offset
  V1 = y_offset

  # Calculate UV at screen origin.
  U = U1 - DUDX*tf32(X1) - DUDY*tf32(Y1)
  V = V1 - DVDX*tf32(X1) - DVDY*tf32(Y1)

  u  = tf.cumsum(tf.fill([h, w], DUDX), 1)
  u += tf.cumsum(tf.fill([h, w], DUDY), 0)
  u += U
  v  = tf.cumsum(tf.fill([h, w], DVDX), 1)
  v += tf.cumsum(tf.fill([h, w], DVDY), 0)
  v += V
  uv = tf.stack([v, u], 2)

  th, tw = h, w
  uv = clamp(iround(wrap(uv, wrap_mode) * [tw, th]), 0, [tw-1, th-1])
  color = tf.gather_nd(img, uv)
  return color

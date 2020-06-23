"""Image Augmentations to be applied to reals and fakes during GAN training, for both GPUs and TPUs.
   Batch functions from Zhao et al 2020's DiffAugment https://github.com/mit-han-lab/data-efficient-gans

   apply to real and fake images in loss function like so:

   reals = RFAugment.augment(reals, policy='zoom in', channels_first=True, mode='tpu')

   TODO: More augs here https://github.com/google/automl/blob/master/efficientdet/aug/autoaugment.py"""

import tensorflow as tf
import os

def augment(batch, policy='', channels_first=True, mode='gpu'):
    if mode == 'gpu':
        if policy:
            if channels_first:
                batch = tf.transpose(batch, [0, 2, 3, 1])
            for p in policy.split(','):
                p = p.replace(" ", "")
                if p in BATCH_AUGMENT_FNS:
                    for f in BATCH_AUGMENT_FNS[p]:
                        batch = f(batch)
                elif p in SINGLE_IMG_FNS:
                    for f in SINGLE_IMG_FNS[p]:
                        batch = tf.map_fn(f, batch)
            if channels_first:
                batch = tf.transpose(batch, [0, 3, 1, 2])
        return batch
    elif mode == 'tpu':
        print('Entering TPU mode of augmentation code')
        if policy:
            if channels_first:
                print('Transposing channels')
                batch = tf.transpose(batch, [0, 2, 3, 1])
            for p in policy.split(','):
                print('POLICY : ', p)
                p = p.replace(" ", "")
                if p in BATCH_AUGMENT_FNS_TPU:
                    for f in BATCH_AUGMENT_FNS_TPU[p]:
                        print('Entering Batch Augmentations')
                        batch = f(batch)
                elif p in SINGLE_IMG_FNS:
                    for f in SINGLE_IMG_FNS_TPU[p]:
                        print('Entering Single Augmentations')
                        batch = tf.map_fn(f, batch)
            if channels_first:
                batch = tf.transpose(batch, [0, 3, 1, 2])
        return batch


alpha_default = 0.1  # set default alpha for spatial augmentations
colour_alpha_default = 0.3  # set default alpha for colour augmentations
alpha_override = float(os.environ.get('SPATIAL_AUGS_ALPHA', '0'))
colour_alpha_override = float(os.environ.get('COLOUR_AUGS_ALPHA', '0'))
if alpha_override > 0:
    if alpha_override >= 1:
        alpha_override = 0.999
    alpha_default = alpha_override
if colour_alpha_override > 0:
    if colour_alpha_override >= 1:
        colour_alpha_override = 0.999
    colour_alpha_default = colour_alpha_override

# ----------------------------------------------------------------------------
# GPU Batch augmentations:


def rand_brightness(x, alpha=colour_alpha_default):
    """
    apply random brightness to image

    :param x: 3-D tensor with a single image.
    :param alpha: Strength of augmentation
    :return: 3-D tensor with a single image.
    """
    magnitude = (tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) - 0.5) * alpha
    x = x + magnitude
    return x


def rand_color(x, alpha=colour_alpha_default):
    """
    apply random colour to image

    :param x: 3-D tensor with a single image.
    :param alpha: Strength of augmentation
    :return:
    """
    magnitude = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) * 2 * alpha
    x_mean = tf.reduce_mean(x, axis=3, keepdims=True)
    x = (x - x_mean) * magnitude + x_mean
    return x


def rand_contrast(x, alpha=colour_alpha_default):
    """
    apply random contrast to image

    :param x: 3-D tensor with a single image.
    :param alpha: Strength of augmentation
    :return: 3-D tensor with a single image.
    """
    magnitude = (tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) + 0.5) * alpha
    x_mean = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
    x = (x - x_mean) * magnitude + x_mean
    return x


def rand_translation(x, ratio=[1, 8]):
    B, H, W = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
    translation_x = tf.random.uniform([B, 1], -(W * ratio[0] // ratio[1]), (W * ratio[0] // ratio[1]) + 1, dtype=tf.int32)
    translation_y = tf.random.uniform([B, 1], -(H * ratio[0] // ratio[1]), (H * ratio[0] // ratio[1]) + 1, dtype=tf.int32)
    grid_x = tf.clip_by_value(tf.expand_dims(tf.range(W, dtype=tf.int32), 0) + translation_x + 1, 0, W + 1)
    grid_y = tf.clip_by_value(tf.expand_dims(tf.range(H, dtype=tf.int32), 0) + translation_y + 1, 0, H + 1)
    x = tf.transpose(tf.gather_nd(tf.pad(tf.transpose(x, [0, 2, 1, 3]), [[0, 0], [1, 1], [0, 0], [0, 0]]), tf.expand_dims(grid_x, -1), batch_dims=1), [0, 2, 1, 3])
    x = tf.gather_nd(tf.pad(x, [[0, 0], [1, 1], [0, 0], [0, 0]]), tf.expand_dims(grid_y, -1), batch_dims=1)
    return x


def rand_cutout(x, ratio=[1, 2]):
    batch_size = tf.shape(x)[0]
    image_size = tf.shape(x)[1:3]
    cutout_size = image_size * ratio[0] // ratio[1]
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


BATCH_AUGMENT_FNS = {
    'color': [rand_brightness, rand_color, rand_contrast],
    'colour': [rand_brightness, rand_color, rand_contrast], # American spelling is a crime
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}

BATCH_AUGMENT_FNS_TPU = {
    'color': [rand_brightness, rand_color, rand_contrast],
    'colour': [rand_brightness, rand_color, rand_contrast],
}

# ----------------------------------------------------------------------------
# GPU single image augmentations


def rand_crop(img, crop_h, crop_w, seed=None):
    """
    Custom implementation of tf.image.random_crop() without all the assertions that break everything.

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


def X_translate(img, alpha=alpha_default, target_image_shape=None, seed=None):
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


def XY_translate(img, alpha=alpha_default, target_image_shape=None, seed=None):
    """
    Random XY translation within TF image with reflection padding
    Args:
      image: 3-D tensor with a single image.
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


def Y_translate(img, alpha=alpha_default, target_image_shape=None, seed=None):
    """
    Random Y translation within TF image with reflection padding
    Args:
      image: 3-D tensor with a single image.
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


def _pad_to_bounding_box(img, offset_height, offset_width, target_height,
                        target_width):
    """Pad `image` with zeros to the specified `height` and `width`.
    Adds `offset_height` rows of zeros on top, `offset_width` columns of
    zeros on the left, and then pads the image on the bottom and right
    with zeros until it has dimensions `target_height`, `target_width`.
    This op does nothing if `offset_*` is zero and the image already has size
    `target_height` by `target_width`.

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

def apply_random_aug(x, seed=None):
    with tf.name_scope('SpatialAugmentations'):
        x.set_shape(x.shape)
        choice = tf.random_uniform([], 0, 6, tf.int32, seed=seed)
        x = tf.cond(tf.reduce_all(tf.equal(choice, tf.constant(0))), lambda: zoom_in(x, seed=seed), lambda: tf.identity(x))
        x = tf.cond(tf.reduce_all(tf.equal(choice, tf.constant(1))), lambda: zoom_out(x, seed=seed), lambda: tf.identity(x))
        x = tf.cond(tf.reduce_all(tf.equal(choice, tf.constant(2))), lambda: X_translate(x, seed=seed), lambda: tf.identity(x))
        x = tf.cond(tf.reduce_all(tf.equal(choice, tf.constant(3))), lambda: Y_translate(x, seed=seed), lambda: tf.identity(x))
        x = tf.cond(tf.reduce_all(tf.equal(choice, tf.constant(4))), lambda: XY_translate(x, seed=seed), lambda: tf.identity(x))
        x = tf.cond(tf.reduce_all(tf.equal(choice, tf.constant(5))), lambda: random_cutout(x, seed=seed), lambda: tf.identity(x))
        return x

SINGLE_IMG_FNS = {
    'zoomin': [zoom_in],
    'zoomout': [zoom_out],
    'xtrans': [X_translate],
    'ytrans': [Y_translate],
    'xytrans': [XY_translate],
    'random': [apply_random_aug]
}

#----------------------------------------------------------------------------
# TPU single image augmentations.

# Okay nothing's working - my only idea is to pad manually. I.E flip image left_right, concat on all sides,
# random crop within large image

def XY_translate_tpu(img, pad_size, y_start = 0, x_start = 0):
    """
    add pad_size padding with mirror reflections on all sides of TF image,
    random crop original image size from padded image

    :param img: 3-D Tensor of shape `[height, width, channels]`
    :param pad_size: int <= height or width - 1
    :param y_start: y start location for crop - int <= height or width - 1
    :param x_start: x start location for crop - int <= height or width - 1
    :return: 3-D Tensor of shape `[height, width, channels]`
    """
    if pad_size == 0:
        return img
    else:
        paddings = [[pad_size, pad_size], [pad_size, pad_size], [0, 0]]
        padded_img = tf.pad(img, paddings, 'CONSTANT')
        return tf.slice(padded_img, [y_start, x_start, 0], img.shape)


def X_translate_tpu(img, pad_size, x_start = 0):
    """
    add pad_size padding with mirror reflections on left and right of TF image,
    random crop original image size from padded image

    :param img: 3-D Tensor of shape `[height, width, channels]`
    :param pad_size: amount of px to pad on each side - int <= height or width - 1
    :param x_start: x start location for crop - int <= height or width - 1
    :return: 3-D Tensor of shape `[height, width, channels]`
    TODO: Mirror padding and randomly selected start for slice doesn't work
    """
    if pad_size == 0:
        return img
    else:
        paddings = [[0, 0], [pad_size, pad_size], [0, 0]]
        padded_img = tf.pad(img, paddings, 'CONSTANT')
        return tf.slice(padded_img, [0, x_start, 0], img.shape)


def Y_translate_tpu(img, pad_size, y_start=0):
    """
    add pad_size padding with mirror reflections on top and bottom of TF image,
    random crop original image size from padded image

    :param img: 3-D Tensor of shape `[height, width, channels]`
    :param pad_size: int <= height or width - 1
    :param y_start: y start location for crop - int <= height or width - 1
    :return: 3-D Tensor of shape `[height, width, channels]`
    """
    if pad_size == 0:
        return img
    else:
        paddings = [[pad_size, pad_size], [0, 0], [0, 0]]
        padded_img = tf.pad(img, paddings, 'CONSTANT')
        # y_start = tf.random_uniform([], 0, (padded_img.shape[0] - img.shape[0]), tf.int32)
        return tf.slice(padded_img, [y_start, 0, 0], img.shape)


def zoom_in_tpu(img, crop_size):
    """
    crop `crop_size x crop_size` section out of a TF image, then resize back to original size.

    :param img: 3-D Tensor of shape `[height, width, channels]`
    :param crop_size: 0 < int <= height or width
    :return: 3-D Tensor of shape `[height, width, channels]`
    """
    # Random crop
    h, w, c = img.shape
    h = tf.cast(
        h, dtype=tf.int32, name='height')
    w = tf.cast(
        w, dtype=tf.int32, name='width')
    begin_h = tf.random_uniform([], 0, h - crop_size, tf.int32)
    begin_w = tf.random_uniform([], 0, w - crop_size, tf.int32)
    img = tf.slice(img, [begin_h, begin_w, 0], [crop_size, crop_size, 3])

    # Resize back to original size
    img = tf.expand_dims(img, 0)
    out = tf.image.resize_bilinear(img, (h, w))
    return tf.squeeze(out)


def zoom_out_tpu(img, pad_size, crop_size):
    """
    add pad_size padding with mirror reflections on each side of TF image,
    crop to crop_size, then resize back to original size.

    :param img: 3-D Tensor of shape `[height, width, channels]`
    :param pad_size: int < height or width - 1
    :param crop_size: int >= height or width
    :return:
    """
    # Pad image
    h, w, c = img.shape
    paddings = [[pad_size, pad_size], [pad_size, pad_size], [0, 0]]
    padded_img = tf.pad(img, paddings, 'REFLECT')

    # Random crop
    if crop_size > padded_img.shape[0]:
        crop_size = padded_img.shape[0]
    elif crop_size < img.shape[0]:
        crop_size = img.shape[0]
    x_start = tf.random_uniform([], 0, (padded_img.shape[1] - crop_size), tf.int32)
    y_start = tf.random_uniform([], 0, (padded_img.shape[0] - crop_size), tf.int32)
    cropped_img = tf.slice(padded_img, [x_start, y_start, 0], [crop_size, crop_size, 3])

    # Resize back to original size
    cropped_img = tf.expand_dims(cropped_img, 0)
    out = tf.image.resize_bilinear(cropped_img, (h, w))
    return tf.squeeze(out)


def pad_to_bounding_box_tpu(image, offset_height, offset_width, target_height, target_width):
    """
    Pad `image` with zeros to the specified `height` and `width`.
    Adds `offset_height` rows of zeros on top, `offset_width` columns of
    zeros on the left, and then pads the image on the bottom and right
    with zeros until it has dimensions `target_height`, `target_width`.
    This op does nothing if `offset_*` is zero and the image already has size
    `target_height` by `target_width`.

    Args:
    image: 3-D Tensor of shape `[height, width, channels]`
    offset_height: Number of rows of zeros to add on top.
    offset_width: Number of columns of zeros to add on the left.
    target_height: Height of output image.
    target_width: Width of output image.
    Returns:3-D float Tensor of shape `[target_height, target_width, channels]`
    """
    h, w, c = image.shape
    after_padding_width = target_width - offset_width - w
    after_padding_height = target_height - offset_height - h
    # Do not pad on the depth dimension.
    paddings = tf.reshape(tf.stack([offset_height, after_padding_height, offset_width, after_padding_width, 0, 0]), [3, 2])
    return tf.pad(image, paddings)


def cutout_tpu(img, cutout_size, y, x):
    """
    Erases a section of the [cutout_size x cutout_size] section of the image at x, y

    :param img: 3-D Tensor of shape `[height, width, channels]`
    :param cutout_size: int < height and width
    :param y: int < height
    :param x: int < width
    :return: 3-D Tensor of shape `[height, width, channels]`
    """
    # get img shape
    shape = img.shape
    h = shape[0]
    w = shape[1]

    erase_area = tf.ones([cutout_size, cutout_size, 3], dtype=tf.float32)

    if erase_area.shape == (0, 0, 3):
        return img
    else:
        mask = 1.0 - pad_to_bounding_box_tpu(erase_area, y, x, h, w)
        erased_img = tf.multiply(img, mask)
        return erased_img


def apply_zoom_in_tpu(x, alpha=alpha_default):
    """
    Random zoom in to TF image, then resize back to original size. Random shapes are impossible on tpus,
    so this operation generates a list of possible pad / crop shapes for each image and makes a selection at random.

    :param x: 3-D Tensor of shape `[height, width, channels]`
    :param alpha: Strength setting of augmentation
    :return: 3-D Tensor of shape `[height, width, channels]`
    """
    # TODO: make alpha setting by constraining range in loop
    x.set_shape(x.shape)
    h = int(x.shape[0])
    # range = img.size*(1-alpha) -> img.size
    min_val = int(h * (1 - alpha))
    choice = tf.random_uniform([], min_val, x.shape[0], tf.int32)
    for crop_size in range(min_val, x.shape[0]):
        x = tf.cond(tf.reduce_all(tf.equal(choice, tf.constant(crop_size))), lambda: zoom_in_tpu(x, crop_size), lambda: tf.identity(x))
    return x


def apply_zoom_out_tpu(x, alpha=alpha_default, range_step = None):
    """
    Random zoom out of TF image with reflection padding. Random shapes are impossible on tpus,
    so this operation generates a list of possible pad / crop shapes for each image and makes a selection at random.

    :param x: 3-D Tensor of shape `[height, width, channels]`
    :param alpha: Strength setting of augmentation
    :param range_step: step in range function to generate pad + crop shapes.
    :return: 3-D Tensor of shape `[height, width, channels]`
    """
    x.set_shape(x.shape)
    h = int(x.shape[0])
    if range_step is None:
        assert h % 32 == 0
        range_step = int(h / 32)
    # crop_range = img.size -> 2*pad_size
    # make list of valid pad sizes
    # the list gets quite large with bigger images and it can slow things down
    # so constrain the size with a step on the range fns
    pad_list = list(range(0, int((h-1)*alpha), range_step))
    # make list of valid crop sizes from each pad size
    crop_list = []
    for p in pad_list:
        valid_crop_list = list(range(h, (h + (2*p) - 1), range_step))
        crop_list.append(valid_crop_list)
    count = sum([len(sublist) for sublist in crop_list])
    # random no corresponds to aug settings to choose
    choice = tf.random_uniform([], 0, count, tf.int32)
    count = 0
    for count_2, p in enumerate(pad_list):
        for c in crop_list[count_2]:
            x = tf.cond(tf.reduce_all(tf.equal(choice, tf.constant(count))), lambda: zoom_out_tpu(x, pad_size = p, crop_size = c), lambda: tf.identity(x))
            count += 1
    return x


def apply_X_translate_tpu(x, alpha=alpha_default, range_step = 4):
    """
    Random X translation within TF image with reflection padding. Random shapes are impossible on tpus,
    so this operation generates a list of possible pad shapes for each image and makes a selection at random.

    :param x: 3-D Tensor of shape `[height, width, channels]`
    :param alpha: Strength setting of augmentation
    :return: 3-D Tensor of shape `[height, width, channels]`
    """
    x.set_shape(x.shape)
    w = int(x.shape[0])
    max_val = int((w - 1) * alpha)
    pad_list = list(range(0, max_val))
    x_loc_list = []
    # crop_range = 0 -> (2*pad_size) - 1
    for p in pad_list:
        valid_crop_list = list(range(0, ((2*p) - 1), range_step))
        x_loc_list.append(valid_crop_list)
    count = sum([len(sublist) for sublist in x_loc_list])
    choice = tf.random_uniform([], 0, count, tf.int32)
    count = 0
    for counter, pad_size in enumerate(pad_list):
        for x_loc in x_loc_list[counter]:
            x = tf.cond(tf.reduce_all(tf.equal(choice, tf.constant(count))), lambda: X_translate_tpu(x, pad_size, x_loc), lambda: tf.identity(x))
            count += 1
    return x


def apply_Y_translate_tpu(x, alpha=alpha_default, range_step = 4):
    """
    Random Y translation within TF image with reflection padding. Random shapes are impossible on tpus,
    so this operation generates a list of possible pad shapes for each image and makes a selection at random.

    :param x: 3-D Tensor of shape `[height, width, channels]`
    :param alpha: Strength setting of augmentation
    :return: 3-D Tensor of shape `[height, width, channels]`
    """
    x.set_shape(x.shape) # set shape of tensor
    h = int(x.shape[0])
    max_val = int((h - 1) * alpha)
    choice = tf.random_uniform([], 0, max_val, tf.int32) # select random augmentation from shape list
    pad_list = list(range(0, max_val))
    y_loc_list = []
    # crop_range = 0 -> (2*pad_size) - 1
    for p in pad_list:
        valid_crop_list = list(range(0, ((2*p) - 1), range_step))
        y_loc_list.append(valid_crop_list)
    count = sum([len(sublist) for sublist in y_loc_list])
    choice = tf.random_uniform([], 0, count, tf.int32)
    count = 0
    # iterate through shape list and perform selected aug
    for counter, pad_size in enumerate(pad_list):
        for y_loc in y_loc_list[counter]:
            x = tf.cond(tf.reduce_all(tf.equal(choice, tf.constant(count))), lambda: Y_translate_tpu(x, pad_size, y_loc), lambda: tf.identity(x))
            count += 1
    return x


def apply_XY_translate_tpu(x, alpha=alpha_default, range_step = 8):
    """
    Random XY translation within TF image with reflection padding. Random shapes are impossible on tpus,
    so this operation generates a list of possible pad / crop shapes for each image and makes a selection at random.

    :param x: 3-D Tensor of shape `[height, width, channels]`
    :param alpha: Strength setting of augmentation
    :return: 3-D Tensor of shape `[height, width, channels]`
    """
    x.set_shape(x.shape) # set shape of tensor
    h = int(x.shape[0])
    max_val = int((h - 1) * alpha)
    pad_list = list(range(0, max_val, range_step))
    crop_loc_list = []
    # crop_range = 0 -> (2*pad_size) - 1
    for p in pad_list:
        x_valid_loc_list = list(range(0, ((2*p) - 1), range_step))
        y_valid_loc_list = x_valid_loc_list
        perms = []
        for y_loc in y_valid_loc_list:
            for x_loc in x_valid_loc_list:
                perms.append([y_loc, x_loc])
        crop_loc_list.append(perms)
    count = sum([len(sublist) for sublist in crop_loc_list])
    choice = tf.random_uniform([], 0, count, tf.int32) # select random augmentation from shape list
    # iterate through shape list and perform selected aug
    for counter, pad_size in enumerate(pad_list):
        for loc in crop_loc_list[counter]:
            x = tf.cond(tf.reduce_all(tf.equal(choice, tf.constant(pad_size))), lambda: XY_translate_tpu(x, pad_size, loc[0], loc[1]), lambda: tf.identity(x))
    return x

def apply_random_cutout_tpu(x, alpha=alpha_default, range_step = None):
    """
    Erases a random section of the image at a set size adjusted by alpha.
    TODO: Optional: pass in a value to p to have a p/1 chance of returning an un-augmented image
    (random tensor shapes are not possible on tpu).

    :param x: 3-D Tensor of shape `[height, width, channels]`
    :param alpha: Strength setting of augmentation - controls the size of the erased section
    :param range_step: step in range function to generate x/y locations for erased section.
    :return: 3-D Tensor of shape `[height, width, channels]`
    """
    x.set_shape(x.shape)
    h = int(x.shape[0])
    cutout_size = int(h * alpha)
    if range_step is None:
      range_step = int(h / 8)
    y_locs = list(range(0, x.shape[0] - cutout_size, range_step))
    x_locs = list(range(0, x.shape[1] - cutout_size, range_step))
    perms = []
    for y_loc in y_locs:
      for x_loc in x_locs:
        perms.append([y_loc, x_loc])
    # random no corresponds to aug settings to choose
    choice = tf.random_uniform([], 0, len(perms), tf.int32)
    for count, perm in enumerate(perms):
        x = tf.cond(tf.reduce_all(tf.equal(choice, tf.constant(count))), lambda: cutout_tpu(x, cutout_size, perm[0], perm[1]), lambda: tf.identity(x))
    return x

def apply_custom_xy_translate(x, alpha=alpha_default, range_step = None):
    """
    Erases a random section of the image at a set size adjusted by alpha.
    TODO: Optional: pass in a value to p to have a p/1 chance of returning an un-augmented image
    (random tensor shapes are not possible on tpu).

    :param x: 3-D Tensor of shape `[height, width, channels]`
    :param alpha: Strength setting of augmentation - controls the size of the erased section
    :param range_step: step in range function to generate x/y locations for erased section.
    :return: 3-D Tensor of shape `[height, width, channels]`
    """
    x.set_shape(x.shape)
    h = int(x.shape[0])
    # cutout_size = int(h * alpha)
    if range_step is None:
      range_step = h // 8
      print(range_step)
    y_locs = list(range(0, (x.shape[0] // 2) - 1, range_step))
    x_locs = list(range(0, (x.shape[1] // 2) - 1, range_step))
    perms = []
    for y_loc in y_locs:
      for x_loc in x_locs:
        perms.append([y_loc, x_loc])
    # random no corresponds to aug settings to choose
    choice = tf.random_uniform([], 0, len(perms), tf.int32)
    for count, perm in enumerate(perms):
        x = tf.cond(tf.reduce_all(tf.equal(choice, tf.constant(count))), lambda: xy_translate_custom(x, perm[0], perm[1]), lambda: tf.identity(x))
    return x

def cut_in_half(img, horizontal=False, left=True, top=True, quarter=True):
  h, w, c = img.shape
  if horizontal:
    if top:
      if quarter:
        return tf.slice(img, [0, 0, 0], [(h//4), w, c])
      else:
        return tf.slice(img, [0, 0, 0], [(h//2), w, c])
    else:
      if quarter:
        return tf.slice(img, [((h//4)*3), 0, 0], [(h//4), w, c])
      else:
        return tf.slice(img, [(h//2), 0, 0], [(h//2), w, c])
  else:
    if left:
      if quarter:
        return tf.slice(img, [0, 0, 0], [h, (w//4), c])
      else:
        return tf.slice(img, [0, 0, 0], [h, (w//4), c])
    else:
      if quarter:
        return tf.slice(img, [0, ((w//4)*3), 0], [h, (w//4), c])
      else:
        return tf.slice(img, [0, (w//2), 0], [h, (w//2), c])


def grid_pad(img, x=False, half=False, quarter=True):
  if quarter:
    half = True
  if x:
    if half:
      img_flippedlr_left = tf.image.flip_left_right(cut_in_half(img, horizontal=False, left=True, quarter=quarter))
      img_flippedlr_right = tf.image.flip_left_right(cut_in_half(img, horizontal=False, left=False, quarter=quarter))
      return tf.concat(values=[img_flippedlr_left, img, img_flippedlr_right], axis=1)
    else:
      img_flippedlr = tf.image.flip_left_right(img)
      return tf.concat(values=[img_flippedlr, img, img_flippedlr], axis=1)
  else:
    if half:
      img_flipped_ud_top = tf.image.flip_up_down(cut_in_half(img, horizontal=True, top=True, quarter=quarter))
      img_flipped_ud_bottom = tf.image.flip_up_down(cut_in_half(img, horizontal=True, top=False, quarter=quarter))
      return tf.concat(values=[img_flipped_ud_top, img, img_flipped_ud_bottom], axis=0)
    else:
      img_flipped_ud = tf.image.flip_up_down(img)
      return tf.concat(values=[img_flipped_ud, img, img_flipped_ud], axis=0)

def mirror_pad_custom(img, quarter=True):
  gp = grid_pad(img, x=True, quarter=True)
  return grid_pad(gp, x=False, quarter=True)

def xy_translate_custom(img, y, x, out_shape=None):
  if out_shape is None:
    out_shape = img.shape
  padded_img = mirror_pad_custom(img)
  padded_shape = padded_img.shape
  if x >= out_shape[1] - 1:
    raise Exception('X value must be < width of image - 1')
  if y >= out_shape[0] - 1:
    raise Exception('Y value must be < height of image - 1')
  return tf.slice(padded_img, [y, x, 0], img.shape)

def apply_random_aug_tpu(x):
    """
    Apply random spatial augmentation to TF image.
    Options: zoom in, zoom out, X/Y/XY translate, random cutout.
    :param x: 3-D Tensor of shape `[height, width, channels]`
    :return: 3-D Tensor of shape `[height, width, channels]`
    """
    x.set_shape(x.shape)
    choice = tf.random_uniform([], 0, 4, tf.int32)
    # x = tf.cond(tf.reduce_all(tf.equal(choice, tf.constant(0))), lambda: apply_zoom_in_tpu(x), lambda: tf.identity(x))
    # x = tf.cond(tf.reduce_all(tf.equal(choice, tf.constant(1))), lambda: apply_zoom_out_tpu(x), lambda: tf.identity(x))
    x = tf.cond(tf.reduce_all(tf.equal(choice, tf.constant(0))), lambda: apply_X_translate_tpu(x), lambda: tf.identity(x))
    x = tf.cond(tf.reduce_all(tf.equal(choice, tf.constant(1))), lambda: apply_Y_translate_tpu(x), lambda: tf.identity(x))
    x = tf.cond(tf.reduce_all(tf.equal(choice, tf.constant(2))), lambda: apply_XY_translate_tpu(x), lambda: tf.identity(x))
    x = tf.cond(tf.reduce_all(tf.equal(choice, tf.constant(3))), lambda: apply_random_cutout_tpu(x), lambda: tf.identity(x))
    return x

SINGLE_IMG_FNS_TPU = {
    'zoomin': [apply_zoom_in_tpu],
    'zoomout': [apply_zoom_out_tpu],
    'xtrans': [apply_X_translate_tpu],
    'ytrans': [apply_Y_translate_tpu],
    'xytrans': [apply_XY_translate_tpu],
    'cutout': [apply_random_cutout_tpu],
    'random': [apply_random_aug_tpu],
    'customxy': [apply_custom_xy_translate]
}

import numpy as np


def data_padding(in_data, padding_size):
    padwidth = padding_size * 2
    padded_data = np.zeros((in_data.shape[0] + padwidth, in_data.shape[1] + padwidth))
    if padding_size > 0:
        padded_data[padding_size:-padding_size, padding_size:-padding_size] = in_data
        return padded_data
    return in_data


def rot180(in_data):
    rotdata = in_data.copy()
    if len(in_data.shape) == 4:
        inchanels = in_data.shape[0]
        outchannels = in_data.shape[1]
        for o in range(outchannels):
            for i in range(inchanels):
                rotdata[i, o] = np.rot90(rotdata[i, o], 2)
        return rotdata
    if len(in_data.shape) == 3:
        channels = in_data.shape[0]
        for j in range(channels):
            rotdata[j] = np.rot90(rotdata[j], 2)
        return rotdata
    if len(in_data.shape) == 2:
        rotdata = np.rot90(rotdata, 2)
    return rotdata


def condense(in_data, padding_size):
    condense_data = np.ndarray(())
    if len(in_data.shape) == 4:
        batchsize, channels, cur_y, cur_x = in_data.shape
        new_y = cur_y - padding_size * 2
        new_x = cur_x - padding_size * 2
        condense_data = np.zeros((batchsize, channels, new_y, new_x))
        for b in range(batchsize):
            for j in range(channels):
                condense_data[b, j] = in_data[b, j, padding_size:cur_y - padding_size,
                                              padding_size:cur_x - padding_size]
    if len(in_data.shape) == 3:
        channels, cur_y, cur_x = in_data.shape
        new_y = cur_y - padding_size * 2
        new_x = cur_x - padding_size * 2
        condense_data = np.zeros((channels, new_y, new_x))
        for j in range(channels):
            condense_data[j] = in_data[j, padding_size:cur_y - padding_size, padding_size:cur_x - padding_size]
    if len(in_data.shape) == 2:
        cur_y, cur_x = in_data.shape
        new_y = cur_y - padding_size * 2
        new_x = cur_x - padding_size * 2
        condense_data = np.zeros((new_y, new_x))
        condense_data = in_data[padding_size:cur_y-padding_size, padding_size:cur_x-padding_size]
    return condense_data


def padding(in_data, padding_size):
    padding_data = np.ndarray(())
    if len(in_data.shape) == 4:
        batchsize, channels, cur_y, cur_x = in_data.shape
        new_y = cur_y + padding_size * 2
        new_x = cur_x + padding_size * 2
        padding_data = np.zeros((batchsize, channels, new_y, new_x))
        for b in range(batchsize):
            for j in range(channels):
                padding_data[b, j] = data_padding(in_data[b, j], padding_size)
    if len(in_data.shape) == 3:
        channels, cur_y, cur_x = in_data.shape
        new_y = cur_y + padding_size * 2
        new_x = cur_x + padding_size * 2
        padding_data = np.zeros((channels, new_y, new_x))
        for j in range(channels):
            padding_data[j] = data_padding(in_data[j], padding_size)
    if len(in_data.shape) == 2:
        cur_y, cur_x = in_data.shape
        new_y = cur_y + padding_size * 2
        new_x = cur_x + padding_size * 2
        padding_data = np.zeros((new_y, new_x))
        padding_data = data_padding(in_data, padding_size)
    return padding_data


def stride_padding(in_data, stride_size):
    strided_padding_in_data = np.ndarray(())
    if len(in_data.shape) == 4:
        batchsize, channels, cur_y, cur_x = in_data.shape
        stride_y = cur_y * stride_size
        stride_x = cur_x * stride_size
        strided_padding_in_data = np.zeros((batchsize, channels, stride_y, stride_x))
        in_data_x = 0
        in_data_y = 0
        for b in range(batchsize):
            for j in range(channels):
                for y in range(stride_y):
                    if y % stride_size == 0:
                        for x in range(stride_x):
                            if x % stride_size == 0:
                                strided_padding_in_data[b, j, y, x] = in_data[b, j, in_data_y, in_data_x]
                                in_data_x += 1
                        in_data_y += 1
                        in_data_x = 0
                in_data_y = 0
    if len(in_data.shape) == 3:
        channels, cur_y, cur_x = in_data.shape
        stride_y = cur_y * stride_size
        stride_x = cur_x * stride_size
        strided_padding_in_data = np.zeros((channels, stride_y, stride_x))
        in_data_x = 0
        in_data_y = 0
        for j in range(channels):
            for y in range(stride_y):
                if y % stride_size == 0:
                    for x in range(stride_x):
                        if x % stride_size == 0:
                            strided_padding_in_data[j, y, x] = in_data[j, in_data_y, in_data_x]
                            in_data_x += 1
                    in_data_y += 1
                    in_data_x = 0
            in_data_y = 0
    if len(in_data.shape) == 2:
        cur_y, cur_x = in_data.shape
        stride_y = cur_y * stride_size
        stride_x = cur_x * stride_size
        strided_padding_in_data = np.zeros((stride_y, stride_x))
        in_data_x = 0
        in_data_y = 0
        for y in range(stride_y):
            if y % stride_size == 0:
                for x in range(stride_x):
                    if x % stride_size == 0:
                        strided_padding_in_data[y, x] = in_data[in_data_y, in_data_x]
                        in_data_x += 1
                in_data_y += 1
                in_data_x = 0
    return strided_padding_in_data


def strided2d(image, kernel, stride):
    strided = np.lib.stride_tricks.as_strided
    s0, s1 = image.strides
    m1, n1 = image.shape
    m2, n2 = kernel.shape
    out_shape = (1 + (m1 - m2) // stride, 1 + (n1 - n2) // stride, m2, n2)
    return strided(image, shape=out_shape, strides=(stride * s0, stride * s1, s0, s1))


def conv2d(image, kernel, stride):
    image_slices = strided2d(image, kernel, stride)
    conv_cols = image_slices.shape[0]
    conv_rows = image_slices.shape[1]
    output_image = np.empty((conv_cols, conv_rows))
    for y in range(conv_cols):
        for x in range(conv_rows):
            output_image[y, x] = np.sum(image_slices[y, x] * kernel)
    return output_image

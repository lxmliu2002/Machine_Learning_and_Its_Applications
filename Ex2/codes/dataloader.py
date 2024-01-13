import struct
import os
import numpy as np

def load_mnist(file_dir, is_images='True'):
    # Read binary data
    bin_file = open(file_dir, 'rb')
    bin_data = bin_file.read()
    bin_file.close()
    # Analysis file header
    if is_images:
        # Read images
        fmt_header = '>iiii'
        magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, 0)
    else:
        # Read labels
        fmt_header = '>ii'
        magic, num_images = struct.unpack_from(fmt_header, bin_data, 0)
        num_rows, num_cols = 1, 1
    data_size = num_images * num_rows * num_cols
    mat_data = struct.unpack_from('>' + str(data_size) + 'B', bin_data, struct.calcsize(fmt_header))
    if is_images:
        mat_data = np.reshape(mat_data, [num_images, num_rows, num_cols])
    else:
        mat_data = np.reshape(mat_data, [num_images])
    print('Load images from %s, number: %d, data shape: %s' % (file_dir, num_images, str(mat_data.shape)))
    return mat_data

def load_data():
    print('Loading MNIST data from files...')

    mnist_dir = "mnist_data/"
    train_data_dir = "train-images.idx3-ubyte"
    train_label_dir = "train-labels.idx1-ubyte"
    test_data_dir = "t10k-images.idx3-ubyte"
    test_label_dir = "t10k-labels.idx1-ubyte"

    train_images = load_mnist(os.path.join(mnist_dir, train_data_dir), True)
    train_labels = load_mnist(os.path.join(mnist_dir, train_label_dir), False)
    test_images = load_mnist(os.path.join(mnist_dir, test_data_dir), True)
    test_labels = load_mnist(os.path.join(mnist_dir, test_label_dir), False)
    return (train_images, train_labels), (test_images, test_labels)
import numpy as np
import struct
import os
import random
import matplotlib.pyplot as plt

# import tqdm


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
    return train_images, train_labels, test_images, test_labels


def normalize(train_images, validation_images, test_images):
    mean_image = np.mean(train_images, axis = 0)
    train_images_normalized = train_images - mean_image
    validation_images_normalized = validation_images - mean_image
    test_images_normalized = test_images - mean_image

    return train_images_normalized, validation_images_normalized, test_images_normalized

def set_data():
    train_images, train_labels, test_images, test_labels = load_data()

    train_images = np.pad(train_images, ((0, 0), (2, 2), (2, 2)))
    test_images = np.pad(test_images, ((0, 0), (2, 2), (2, 2)))

    train_images = train_images.astype(np.float32).reshape(-1, 1, train_images.shape[1], train_images.shape[2])
    test_images = test_images.astype(np.float32).reshape(-1, 1, test_images.shape[1], test_images.shape[2])
    validation_images = train_images[-2000:]
    validation_labels = train_labels[-2000:]
    train_images = train_images[:-2000]
    train_labels = train_labels[:-2000]

    train_images, validation_images, test_images = normalize(train_images, validation_images, test_images)

    return {
        "train_image": train_images,
        "train_label": train_labels,
        "val_image": validation_images,
        "val_label": validation_labels,
        "test_image": test_images,
        "test_label": test_labels,
    }

def get_batch(X, Y, batch_size):
    N = len(X)

    # Generate random indices without replacement
    indices = np.random.choice(N, batch_size, replace=False)

    # Use the indices to get a batch of samples
    batch_X = X[indices]
    batch_Y = Y[indices]

    return batch_X, batch_Y


def plot_result(loss_list, acc_list):
    # 分别绘制 loss 和 accuracy 曲线 并保存图片
    plt.figure()
    plt.plot(loss_list)
    plt.title("loss_lxmliu2002")
    # plt.xlabel("iteration")
    plt.ylabel("loss")
    # plt.savefig('loss.png')
    plt.figure()
    plt.plot(acc_list)
    plt.title("acc_lxmliu2002")
    # plt.xlabel("iteration")
    plt.ylabel("acc")
    # plt.savefig('acc.png')

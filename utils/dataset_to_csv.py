import sys
import os
import cv2
import urllib
import tarfile

import imageio
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image
import tensorflow as tf
import numpy as np
import csv
from io import BytesIO  # "import StringIO" directly in python2
import matplotlib.image as mpimg

cifarClassName = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def download_and_unpack(dst_path):
    # Download CIFAR-10 dataset from https://www.cs.toronto.edu/~kriz/cifar.html
    print("Downloading dataset")
    urllib.request.urlretrieve('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', 'cifar-10-python.tar.gz')
    with tarfile.open('cifar-10-python.tar.gz') as tar:
        tar.extractall(dst_path)
    os.remove("cifar-10-python.tar.gz")


def get_dataset_rgb(path):
    # For each training batch read and concatenate into one big numpy array
    # You may find the format of CIFAR-10 in Data Layout section https://www.cs.toronto.edu/~kriz/cifar.html
    # Train Data
    cifarImages = np.empty((0, 3072), dtype=np.uint8)
    cifarLabels = np.empty((0,), dtype=np.uint8)
    for batchNo in range(1, 6):
        dataDict = unpickle(path + '/data_batch_' + str(batchNo))
        cifarImages = np.vstack((cifarImages, dataDict[b'data']))
        cifarLabels = np.hstack((cifarLabels, dataDict[b'labels']))

    train_dataset = cifarImages, cifarLabels

    # Test Data
    dataDict = unpickle(path + '/test_batch')
    cifarTestImages = dataDict[b'data']
    cifarTestLabels = np.array(dataDict[b'labels'])
    test_dataset = cifarTestImages, cifarTestLabels
    return train_dataset, test_dataset


def convert_rgb_to_grayscale(rgb_dataset):
    # Convert all train images to grayscale
    train_dataset_rgb, test_dataset_rgb = rgb_dataset
    train_images_rgb, train_labels = train_dataset_rgb
    test_images_rgb, test_labels = test_dataset_rgb

    cifarImages_rgb_train = []
    for i in range(len(train_images_rgb)):
        cifarImages_rgb_train.append(train_images_rgb[i, :].reshape(3, 32, 32).transpose(1, 2, 0))
    train_images_gray = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in cifarImages_rgb_train])

    cifarImages_rgb_test = []
    for i in range(len(test_images_rgb)):
        cifarImages_rgb_test.append(test_images_rgb[i, :].reshape(3, 32, 32).transpose(1, 2, 0))
    test_images_gray = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in cifarImages_rgb_test])

    return (train_images_gray, train_labels), (test_images_gray, test_labels)


def dataset_gray_to_csv(dataset, dst):
    header = []
    for i in range(32 * 32):
        header.append("Px " + str(i))
    header.append("class")

    images, labels = dataset

    i = 0
    with open(dst, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for image, label in zip(images, labels):
            value = image.flatten()
            value = np.append(value, cifarClassName[label])
            writer.writerow(value)
            i = i + 1
            print('\r' + "Progress: {:4}/{:4} images | {:.2f}% ".format(i, len(images),
                                                                        (float(i) / float(len(images)) * 100.0)),
                  end='')
    print("\n")


def dataset_RGB_to_csv(dataset, dst, quality=100):
    header = []
    for i in range(32 * 32):
        header.append("Px R " + str(i))
        header.append("Px G " + str(i))
        header.append("Px B " + str(i))
    header.append("class")

    i = 0
    image_list = []
    images, labels = dataset

    with open(dst, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for image, label in zip(images, labels):
            image_rgb = image.reshape(3, 32, 32).transpose(1, 2, 0)  # .flatten()
            image_rgb = Image.fromarray(image_rgb)

            buffer = BytesIO()
            image_rgb.save(buffer, "JPEG", quality=quality)
            image_compressed = Image.open(buffer)
            value = np.array(image_compressed).flatten()
            value = np.append(value, cifarClassName[label])
            writer.writerow(value)
            i = i + 1
            print('\r' + "Progress: {:4}/{:4} images | {:.2f}% ".format(i, len(images),
                                                                        (float(i) / float(len(images)) * 100.0)),
                  end='')
    print("\n")


def jpeg_compress(image, quality):
    buf = BytesIO()
    imageio.imwrite(buf, image, format='jpg', quality=quality)
    s = buf.getbuffer()
    return imageio.imread(s, format='jpg')


def convolve_dataset_2d(gray_dataset, kernel_, padding=0, strides=1):
    kernel = cv2.flip(kernel_, -1)

    train_dataset, test_dataset = gray_dataset
    train_images, train_labels = train_dataset
    test_images, test_labels = test_dataset

    train_images_gray = []
    for image in train_images:
        train_images_gray.append(cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_CONSTANT))
    # train_images_gray = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in cifarImages_rgb_train])

    test_images_gray = []
    for image in test_images:
        test_images_gray.append(cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_CONSTANT))

    return (train_images_gray, train_labels), (test_images_gray, test_labels)


def generate_compression_overview(dataset, qualitys):
    images, labels = dataset
    compressed_images = []
    # take first image
    image_rgb = images[2].reshape(3, 32, 32).transpose(1, 2, 0)
    im1 = Image.fromarray(image_rgb)
    dim = 3
    plt.figure(figsize=(8, 8))
    for i, quality in enumerate(qualitys):
        # here, we create an empty string buffer
        buffer = BytesIO()
        im1.save(buffer, "JPEG", quality=quality)
        size = buffer.tell()
        plt.subplot(dim, dim, i + 1)
        plt.title("q: " + str(quality) + " s: " + str(size) + "bytes")
        plt.axis('off')
        img = mpimg.imread(buffer, format='jpeg')
        plt.imshow(img)
        with open("./quality_" + str(quality) + ".jpg", "wb") as file:
            file.write(buffer.getvalue())
        buffer.seek(0)
    plt.savefig("./matrix.jpg")
    plt.show()


if __name__ == "__main__":
    dataset_path = "../datasets/"
    cifar_path = dataset_path + "cifar-10-batches-py"

    if not os.path.isdir(cifar_path):
        download_and_unpack(dst_path=cifar_path)

    dataset_rgb = get_dataset_rgb(cifar_path)  # (train_dataset, test_dataset)
    dataset_RGB_train, dataset_RGB_test = dataset_rgb

    quality_list = [1, 5, 10, 15, 20, 25, 50, 75, 100]
    # generate_compression_overview(dataset_RGB_train, quality_list)

    for quality in quality_list:
        path_train = dataset_path + "cifar10_RGB_quality_" + str(quality) + "_train.csv"
        path_test = dataset_path + "cifar10_RGB_quality_" + str(quality) + "_test.csv"
        if not os.path.isfile(path_train):
            dataset_RGB_to_csv(dataset=dataset_RGB_train, dst=path_train, quality=quality)
        if not os.path.isfile(path_test):
            dataset_RGB_to_csv(dataset=dataset_RGB_test, dst=path_test, quality=quality)

    exit()

    dataset_gray_train, dataset_gray_test = convert_rgb_to_grayscale(dataset_rgb)

    if not os.path.isfile(dataset_path + "cifar10_gray_train.csv"):
        dataset_gray_to_csv(dataset=dataset_gray_train, dst=dataset_path + "cifar10_gray_train.csv")

    if not os.path.isfile(dataset_path + "cifar10_gray_test.csv"):
        dataset_gray_to_csv(dataset=dataset_gray_test, dst=dataset_path + "cifar10_gray_test.csv")

    kernel = np.asarray([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.uint8)
    dataset_gray_train_conv2D, dataset_gray_test_conv2D = convolve_dataset_2d((dataset_gray_train, dataset_gray_test),
                                                                              kernel)
    # dataset_gray_to_csv(dataset=dataset_gray_train_conv2D, dst=dataset_path + "cifar10_gray_train_conv2d.csv")
    dataset_gray_to_csv(dataset=dataset_gray_test_conv2D, dst=dataset_path + "cifar10_gray_test_conv2d.csv")

import csv
import argparse
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import random
import h5py
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda, Conv2D, MaxPooling2D, Cropping2D, Dropout


# # %matplotlib inline


def read_rgb_image(image_file):
    img = cv.imread(image_file)
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)


def image_selection_random_from_left_right(left_img_f, center_img_f, right_img_f, steering, correction):
    rand_val = np.random.choice(2)

    yield read_rgb_image(center_img_f), steering
    if (rand_val % 2) == 0:
        yield read_rgb_image(left_img_f), steering + correction
    else:
        yield read_rgb_image(right_img_f), steering - correction


def read_data(driving_log_dir, header=False):
    lines = []
    driving_log = driving_log_dir + "/driving_log.csv"

    with open(driving_log) as csvfile:
        reader = csv.reader(csvfile)

        for line in reader:
            if header:
                header = False
                continue
            lines.append(line)
    return lines


def get_training_data(driving_log_dir, base_dir='', header=False, correction=0.20, image_selection=""):
    lines = read_data(driving_log_dir, header)
    images = []
    steerings = []
    for line in lines:
        center_img_f = base_dir + line[0]
        left_img_f = base_dir + line[1]
        right_img_f = base_dir + line[2]
        steering_ang = float(line[3])

        if image_selection == "":
            for (img, ang) in image_selection_random_from_left_right(left_img_f, center_img_f, right_img_f,
                                                                     steering_ang, correction):
                images.append(img)
                steerings.append(ang)

    return images, steerings


def augmnet_training_data(images, steerings):
    cur_len = len(images)
    for i in range(cur_len):
        images.append(np.fliplr(images[i]))
        steerings.append(-steerings[i])

    return images, steerings


def Nvidia_model(in_shape, dropout):
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=in_shape))
    model.add(Cropping2D(cropping=((50, 20), (0, 0))))

    model.add(Conv2D(24, (5, 5), padding='valid', activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(36, (5, 5), padding='valid', activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(48, (5, 5), padding='valid', activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='valid', activation='tanh'))

    model.add(Conv2D(64, (3, 3), padding='valid', activation='tanh'))

    model.add(Flatten())

    model.add(Dense(100))
    model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.summary()

    return model


def train_model(model_name, images, steerings, dropout=0.20, epochs=10):
    model = None
    if model_name == "nvidia":
        model = Nvidia_model(images[0].shape, dropout)

    model.compile(optimizer='adam', loss='mse')

    model.fit(images, steerings, validation_split=0.10, batch_size=64, shuffle=True, epochs=epochs)

    model.save(model_name + '.h5')


def main():
    parser = argparse.ArgumentParser(description='self Driving')

    parser.add_argument('driving_log_dir', type=str, help='driving log directory.')

    parser.add_argument('model_name', type=str, help='model name')

    args = parser.parse_args()

    print(args.driving_log_dir)

    images, steerings = get_training_data(args.driving_log_dir)

    images, steerings = augmnet_training_data(images, steerings)

    train_model(args.model_name, np.array(images), np.array(steerings))


if __name__ == '__main__':
    main()

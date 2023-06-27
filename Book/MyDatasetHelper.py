import numpy as np
import os
import cv2
import pickle
import copy


class MyDatasetHelper:

    def __init__(self, dataset_path, class_names, image_size,
                 train_test_split=0.2, shuffle_data=True):
        self.dataset_path = dataset_path
        self.class_names = class_names
        self.image_size = image_size
        self.train_test_split = train_test_split
        self.shuffle_data = shuffle_data
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []

    def load_dataset(self):
        print("Loading dataset...")
        for class_index, class_name in enumerate(self.class_names):
            path = os.path.join(self.dataset_path, str(class_index))
            files = os.listdir(path)
            for file_name in files:
                file_path = os.path.join(path, file_name)
                image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                image = cv2.resize(image, self.image_size)
                self.X_train.append(image)
                self.y_train.append(class_index)

        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)

        if self.shuffle_data:
            self.shuffle()

        self.X_test = copy.deepcopy(
            self.X_train[int(len(self.X_train) * (1 - self.train_test_split)):])
        self.y_test = copy.deepcopy(
            self.y_train[int(len(self.y_train) * (1 - self.train_test_split)):])
        self.X_train = copy.deepcopy(
            self.X_train[:int(len(self.X_train) * (1 - self.train_test_split))])
        self.y_train = copy.deepcopy(
            self.y_train[:int(len(self.y_train) * (1 - self.train_test_split))])

        print("Dataset loaded")

    def shuffle(self):
        print("Shuffling dataset...")
        permutation = np.random.permutation(len(self.X_train))
        self.X_train = self.X_train[permutation]
        self.y_train = self.y_train[permutation]
        print("Dataset shuffled")

    def normalize(self):
        print("Normalizing dataset...")
        self.X_train = (self.X_train.astype(np.float32) - 127.5) / 127.5
        self.X_test = (self.X_test.astype(np.float32) - 127.5) / 127.5
        print("Dataset normalized")

    def invert_image_colors(self):
        print("Inverting image colors...")
        self.X_train = 255 - self.X_train
        self.X_test = 255 - self.X_test
        print("Image colors inverted")

    def reshape_to_data(self):
        print("Reshaping dataset...")
        self.X_train = self.X_train.reshape(len(self.X_train), -1)
        self.X_test = self.X_test.reshape(len(self.X_test), -1)
        print("Dataset reshaped")

    def reshape_to_images(self):
        print("Reshaping dataset...")
        self.X_train = self.X_train.reshape(
            len(self.X_train), self.image_size[0], self.image_size[1])
        self.X_test = self.X_test.reshape(
            len(self.X_test), self.image_size[0], self.image_size[1])
        print("Dataset reshaped")

    def print_dataset_info(self):
        print("Dataset info:")
        print("X_train shape:", self.X_train.shape)
        print("y_train shape:", self.y_train.shape)
        print("X_test shape:", self.X_test.shape)
        print("y_test shape:", self.y_test.shape)

    def save_dataset(self, file_name):
        print("Saving dataset...")
        with open(file_name, 'wb') as f:
            pickle.dump([self.X_train, self.y_train,
                        self.X_test, self.y_test], f)
        print("Dataset saved")

    def load_saved_dataset(self, file_name):
        print("Loading saved dataset...")
        with open(file_name, 'rb') as f:
            self.X_train, self.y_train, self.X_test, self.y_test = pickle.load(
                f)
        print("Saved dataset loaded")


    @staticmethod
    def load_saved_dataset_static(file_name):
        print("Loading saved dataset...")
        with open(file_name, 'rb') as f:
            X_train, y_train, X_test, y_test = pickle.load(f)
        print("Saved dataset loaded")
        return X_train, y_train, X_test, y_test
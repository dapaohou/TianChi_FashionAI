'''
# 天池比赛，数据读取部分，详细说明及示例代码见底部
# 20180313做了如下的修改：
# 1.类构造时的参数img_size可以控制图片缩放，如img_size=224则输出图片尺寸为[None, 224, 224, 3]，
#   相应的landmarks不发生改变,因为网络可以学习到变换系数
# 2.返回的图片为float，返回的landmarks/visibilities为int，需要在tensorflow以int形式接收，然后转换
# 3.训练集与warmup集的csv名称不同，已改正
'''

import numpy as np
import csv
import os
import skimage.io
import skimage.transform


class TianChiDateReader(object):
    def __init__(self, data_dir, img_size=512, landmarks_num=24, train_size=0.8):
        self.train_size = train_size    # (1-train_size) for validation, (train_size) for training
        self.img_size = img_size        # the image is img_size*img_size*3
        self.landmarks_num = landmarks_num    # the number of landmarks
        self.data_dir = data_dir    # the data dir whose subfolders are "Annotations" and "images"
        self.images_path = []
        self.landmarks_x = []
        self.landmarks_y = []
        self.visibilities = []

        self.read_csv()
        self.shuffle_data()
        self.merge_landmark()
        print("TianChiDataReader: get {} images successfully...".format(self.landmarks_merged.shape[0]))

    def read_csv(self):
        csv_file_path = os.path.join(self.data_dir, 'Annotations/train.csv')
        with open(csv_file_path, encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                if csv_reader.line_num == 1:    # ignore the first line
                    continue
                for ii, data in enumerate(row):     # save useful data to lists
                    if ii == 0:
                        self.images_path.append(data)
                    elif 2 <= ii < self.landmarks_num+2:
                        landmark_and_visibility = data.split("_")
                        for jj, data2 in enumerate(landmark_and_visibility):
                            if jj == 0:
                                self.landmarks_x.append(int(data2))
                            elif jj == 1:
                                self.landmarks_y.append(int(data2))
                            elif jj == 2:
                                self.visibilities.append(int(data2))
                    else:
                        continue

        self.images_path = np.array(self.images_path).reshape(-1, 1)
        self.landmarks_x = np.array(self.landmarks_x).reshape(-1, self.landmarks_num)
        self.landmarks_y = np.array(self.landmarks_y).reshape(-1, self.landmarks_num)
        self.visibilities = np.array(self.visibilities).reshape(-1, self.landmarks_num)
        # print(self.image_paths)
        # print(self.landmarks_x)
        # print(self.landmarks_y)
        # print(self.visibilities)

    def shuffle_data(self):
        permutation = np.random.permutation(self.images_path.shape[0])
        self.images_path = self.images_path[permutation, :]
        self.landmarks_x = self.landmarks_x[permutation, :]
        self.landmarks_y = self.landmarks_y[permutation, :]
        self.visibilities = self.visibilities[permutation, :]
        # print(self.image_paths[10])
        # print(self.landmarks_x[10])
        # print(self.landmarks_y[10])
        # print(self.visibilities[10])

    # merge to: landmark:[x1, x2, ... ,y1, y2, ...]  visibility:[x1_vis, x2_vis, ... , y1_vis, y2_vis, ...]
    def merge_landmark(self):
        self.landmarks_merged = np.hstack((self.landmarks_x, self.landmarks_y))
        self.visibilities_merged = np.hstack((self.visibilities, self.visibilities))

    def load_images(self, images_path):
        images_batch = []
        for path in images_path:
            image = skimage.io.imread(os.path.join(self.data_dir, path))
            if(self.img_size != 512):
                image = skimage.transform.resize(image, (self.img_size, self.img_size))
            images_batch.append(image.reshape(1, self.img_size, self.img_size, 3))
        images_batch = np.array(images_batch).reshape(-1, self.img_size, self.img_size, 3)
        return images_batch

    def get_train_data(self, batch_size):
        num_train = int(self.landmarks_merged.shape[0] * self.train_size)
        num_batch = num_train // batch_size

        for ii in range(0, num_batch * batch_size, batch_size):
            if ii != (num_batch - 1) * batch_size:      # not the last batch
                images_path_batch = self.images_path[ii: ii + batch_size, 0]
                landmarks_batch = self.landmarks_merged[ii: ii+batch_size, :]
                visibilities_batch = self.visibilities_merged[ii: ii+batch_size, :]
            else:   # the last batch
                images_path_batch = self.images_path[ii:num_train, 0]
                landmarks_batch = self.landmarks_merged[ii:num_train, :]
                visibilities_batch = self.visibilities_merged[ii:num_train, :]
            images_batch = self.load_images(images_path_batch)

            yield images_batch, landmarks_batch, visibilities_batch

    def get_validation_data(self):
        num_train = int(self.landmarks_merged.shape[0] * self.train_size)

        images_path_batch = self.images_path[num_train:, 0]
        landmarks_batch = self.landmarks_merged[num_train:, :]
        visibilities_batch = self.visibilities_merged[num_train:, :]
        images_batch = self.load_images(images_path_batch)

        return images_batch, landmarks_batch, visibilities_batch

    def get_validation_data_iter(self, batch_size):
        num_train = int(self.landmarks_merged.shape[0] * self.train_size)
        num_batch = (self.landmarks_merged.shape[0] - num_train) // batch_size
        for ii in range(num_train, self.landmarks_merged.shape[0], batch_size):
            if (ii + batch_size) <= self.landmarks_merged.shape[0]:
                images_path_batch = self.images_path[ii: ii + batch_size, 0]
                landmarks_batch = self.landmarks_merged[ii: ii + batch_size, :]
                visibilities_batch = self.visibilities_merged[ii: ii + batch_size, :]
            else:  # the last batch
                images_path_batch = self.images_path[ii:, 0]
                landmarks_batch = self.landmarks_merged[ii:, :]
                visibilities_batch = self.visibilities_merged[ii:, :]

            images_batch = self.load_images(images_path_batch)

            yield images_batch, landmarks_batch, visibilities_batch


if __name__ == "__main__":
    ''' 用法说明：
    # 0.首先确保本文件与主文件在同一目录下，主文件添加 import TianChiDateReader
    #
    # 1.声明一个“天池数据读取器”，"../train_modified"为我电脑上的数据路径，train_size=0.8代表有80%的数据用于train,20%数据用于validate
    # tr = TianChiDateReader("../train_modified", img_size=512, train_size=0.8)
    #
    # 2.用于获取训练集数据的函数get_train_data(batch_size)是迭代器，用for的形式来调用。迭代器会在每次循环中，给出新一组batch的数据
    # for images, landmarks, visibilities in tr.get_train_data(batch_size=1000):
    #     print(images.shape)
    #     print(landmarks.shape)
    #     print(visibilities.shape)
    # 
    # 3.用于获取验证集数据的函数get_validation_data()是普通函数，以普通方式调用
    # images, landmarks, visibilities = tr.get_validation_data()
    # print(images.shape)
    # print(landmarks.shape)
    # print(visibilities.shape)
    #
    # 4.当验证集数据量很大，无法一次性读入时，可以使用迭代器函数get_validation_data_iter(batch_size)
    # for images, landmarks, visibilities in tr.get_validation_data_iter(batch_size=1000):
    #     print(images.shape)
    #     print(landmarks.shape)
    #     print(visibilities.shape)
    # 注：给出的batch形如：image:[None, 512, 512, 3], landmark:[x1,x2,...,y1,y2,...], visibility:[x1_vis,x2_vis,...,y1_vis,y2_vis,...]
    '''



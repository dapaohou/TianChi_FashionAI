"""
# 天池比赛，测试集读取及结果写入部分，详细说明及示例代码见底部
# 20180313做了如下的修改：
# 1.类构造时的参数img_size可以控制图片缩放，如img_size=224则输出图片尺寸为[None, 224, 224, 3]，
#   相应的landmarks不发生改变,因为网络可以学习到全局变换系数
# 2.加载测试集图片时，返回的图片为float，需要在tensorflow以int形式接收，然后转换
# 3.当写入结果的时候，网络输出的landmarks应该转换为int型，再调用函数保存
"""

import numpy as np
import skimage.io
import csv
import os
import time


class TianChiTestRecorder(object):
    def __init__(self, data_dir, batch_size, img_size=512, landmarks_num=24):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.landmarks_num = landmarks_num
        self.images_path = []
        self.images_class = []

        self.read_csv()  # get image's path
        self.num_batch = self.images_path.shape[0] // batch_size

        self.result_x = np.zeros((self.images_path.shape[0], self.landmarks_num), dtype=np.int64)
        self.result_y = np.zeros((self.images_path.shape[0], self.landmarks_num), dtype=np.int64)
        print("TianChiTester: get {} images, divide to {} batches...".format(self.images_path.shape[0], self.num_batch))

    def read_csv(self):
        csv_file_path = os.path.join(self.data_dir, "test.csv")
        with open(csv_file_path, encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                if csv_reader.line_num == 1:  # ignore the first line
                    continue
                for ii, data in enumerate(row):  # get image's path to lists
                    if ii == 0:
                        self.images_path.append(data)
                    elif ii == 1:
                        self.images_class.append(data)
                    else:
                        continue
        self.images_path = np.array(self.images_path).reshape(-1, 1)
        self.images_class = np.array(self.images_class).reshape(-1, 1)
        # print(self.images_path)
        # print(self.images_class)

    def get_num_batch_and_num_image(self):
        return self.num_batch, self.images_path.shape[0]

    def load_images(self, index_start, index_end):
        images_batch = []
        images_path = self.images_path[index_start:index_end, :]
        for path in images_path[:, 0]:
            image = skimage.io.imread(os.path.join(self.data_dir, path))
            if (self.img_size != 512):
                image = skimage.transform.resize(image, (self.img_size, self.img_size))
            images_batch.append(image.reshape(1, self.img_size, self.img_size, 3))
        images_batch = np.array(images_batch).reshape(-1, self.img_size, self.img_size, 3)
        return images_batch

    def get_batch(self, batch_index):      # batch_index: from 0 to (num_batch-1)
        if batch_index != self.num_batch-1:
            index_start = 0 + batch_index * self.batch_size
            index_end = index_start + self.batch_size
        else:       # the last batch
            index_start = 0 + batch_index * self.batch_size
            index_end = self.images_path.shape[0]
        return self.load_images(index_start, index_end)

    def record_data(self, data, index_start, index_end):    # record data to list
        assert data.shape[0] == index_end - index_start
        self.result_x[index_start:index_end, :] = data[:, :data.shape[1]//2]
        self.result_y[index_start:index_end, :] = data[:, data.shape[1]//2:]
        # print(self.result_x)
        # print(self.result_y)

    def record_batch(self, data, batch_index):
        if batch_index != self.num_batch-1:
            index_start = 0 + batch_index * self.batch_size
            index_end = index_start + self.batch_size
        else:
            index_start = 0 + batch_index * self.batch_size
            index_end = self.images_path.shape[0]
        self.record_data(data, index_start, index_end)

    def write_csv_file(self, simple_visibility=True):       # write recorded data to csv file
        target_file_name = time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time())) + ".csv"  # new file
        target_path = os.path.join(self.data_dir, target_file_name)
        if simple_visibility == False:
            pass    # visibility暂不考虑
        with open(target_path, 'xt', encoding='utf-8') as f:
            f.write("image_id,image_category,neckline_left,neckline_right,center_front,shoulder_left,shoulder_right,"
                    "armpit_left,armpit_right,waistline_left,waistline_right,cuff_left_in,cuff_left_out,cuff_right_in,"
                    "cuff_right_out,top_hem_left,top_hem_right,waistband_left,waistband_right,"
                    "hemline_left,hemline_right,crotch,bottom_left_in,bottom_left_out,bottom_right_in,bottom_right_out")
            for ii in range(self.images_path.shape[0]):
                f.write("\r\n")
                f.write(str(self.images_path[ii][0]) + "," + str(self.images_class[ii][0]))
                for jj in range(self.landmarks_num):
                    text = str(self.result_x[ii][jj]) + "_" + str(self.result_y[ii][jj]) + "_1"
                    f.write("," + text)


if __name__ == "__main__":
    """使用说明：
    # 0.首先确保本文件与主文件在同一目录下，主文件添加 from TianChiTestRecorder import TianChiTestRecorder
    # 
    # 1.声明一个“天池测试集记录器”，"../test_modified"是我电脑上测试集的地址。因测试集图片较多，也使用batch的方式给出图片
    # tt = TianChiTestRecorder("../test_modified", batch_size=1000, img_size=512)
    # 
    # 2.使用函数get_num_batch_and_num_image()测试集图片数量及batch数目
    # num_batch, num_image = tt.get_num_batch_and_num_image()
    # print("共得到{}张图片，分为{}个batch".format(num_image, num_batch))
    # 
    # 3.使用函数get_batch(batch_index)来获取每个batch的图片，这里batch_index的范围是0 ~ (num_batch-1)
    # for index in range(num_batch):
    #     images = tt.get_batch(batch_index=index)
    #     print(images.shape)
    #     
    #     这里的代码用来模拟深度网络的输出，输出维度为[None, 24*2]
    #     data_x = np.ones((images.shape[0], 24), dtype=np.int64) * (index+1)
    #     data_y = np.ones((images.shape[0], 24), dtype=np.int64) * (index+1) * 2
    #     data = np.hstack((data_x, data_y))
    #     
    #     4.使用函数record_batch(data, batch_index)解析网络的输出，并记录到内存
    #     tt.record_batch(data, batch_index=index)
    # 
    # 5.测试集数据全部跑完后，使用函数write_csv_file把内存中的结果写入csv，注意不要遗忘这一步
    # tt.write_csv_file()
    # 
    # 以上为一般测试过程需要用到的函数，另外，为了便于灵活调用，还提供以下函数：
    # load_images(self, index_start, index_end)  从[index_start, index_end)加载测试集图片，注意区间前闭后开
    # record_data(self, data, index_start, index_end)  解析网络在[index_start, index_end)的输出并写入内存，注意区间前闭后开
    # 
    # 注：网络的输出形式应为：[x1,x2,x3,...,y1,y2,y3,...]
"""



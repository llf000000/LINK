import torch
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np

class DatasetProcessing(Dataset):
    def __init__(self, data_path, img_filename, transform=None):
        self.img_path = data_path
        self.transform = transform
        # reading img file from file
        img_filepath = img_filename
        fp = open(img_filepath, 'r') # 读取的txt文件

        self.img_filename = [] # 图像文件名
        self.labels = [] # 标签 病变等级
        # self.lesions = [] # 病变信息 即痤疮个数
        for line in fp.readlines():
            filename, label, lesion = line.split()
            self.img_filename.append(filename)
            self.labels.append(int(label))
            # self.lesions.append(int(lesion))
        # self.img_filename = [x.strip() for x in fp]
        fp.close()
        self.img_filename = np.array(self.img_filename)
        self.labels = np.array(self.labels)#.reshape(-1, 1)
        # self.lesions = np.array(self.lesions)#.reshape(-1, 1)

        if 'NNEW_trainval' in img_filename:
            ratio = 1.0#0.1
            import random
            # random.seed(42)
            indexes = []
            for i in range(4):
                index = random.sample(list(np.where(self.labels == i)[0]), int(len(np.where(self.labels == i)[0]) * ratio))
                indexes.extend(index)
            self.img_filename = self.img_filename[indexes]
            self.labels = self.labels[indexes]
            # self.lesions = self.lesions[indexes]

        # reading labels from file
        # label_filepath = os.path.join(data_path, label_filename)
        # labels = np.loadtxt(label_filepath, dtype=np.int64)

        # self.label = labels

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        # name = self.img_filename[index]
        label = torch.from_numpy(np.array(self.labels[index]))
        # lesion = torch.from_numpy(np.array(self.lesions[index]))
        # return img, label, lesion
        return img, label

    def __len__(self):
        return len(self.img_filename)
    @property
    def targets(self):
        return self.labels

'''
有序组织：在代码中，通过循环 for i in range(4):，对每个标签值（0、1、2、3）分别进行处理。对于每个标签值 i，它会找到所有标签为 i 的样本的索引，并根据指定的比例 ratio 进行随机抽样。因为这个过程是按标签值的顺序执行的（从0开始，逐个递增），所以最后收集的索引列表 indexes 会保持这个顺序：首先是所有选中的标签为0的样本索引，接着是标签为1的样本索引，依此类推。
列表更新：在完成对所有标签值的处理后，原始的 self.img_filename、self.labels 和 self.lesions 被更新为仅包含通过随机抽样选中的样本。因为这些更新是基于有序的 indexes 列表进行的，所以更新后的列表中样本的顺序会与 indexes 中的顺序相匹配。
结果解释：因此，最终的 img、labels 和 lesions 是按照标签顺序有序组织的，其中第一部分包含所有选中的标签为0的样本，然后是标签为1的样本，依此类推，直到最后一部分的标签为3的样本。
'''
'''
为什么要这样组织,可能是因为交叉验证
'''
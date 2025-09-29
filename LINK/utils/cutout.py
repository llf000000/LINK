import torch
import numpy as np


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class ConditionalTransform:
    def __init__(self, transform, num_per_cls_dict):
        self.transform = transform
        self.num_per_cls_dict = num_per_cls_dict
        self.n_holes_dict = {}

    def __call__(self, img, label):
        total_samples = sum(self.num_per_cls_dict.values())
        cls_num_list = list(self.num_per_cls_dict.values())
        per_cls_weights = 1.0 / np.array(cls_num_list)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(torch.device('cpu'))  # 假设我们在 CPU 上运行

        # 计算 n_holes 的数量
        n_holes = 1 + int(per_cls_weights[label] * 3)  # 确保 n_holes 在 1 到 4 之间
        n_holes = min(4, max(1, n_holes))

        # 保存每个类别的 n_holes 数量
        self.n_holes_dict[label] = n_holes

        img = self.transform(img)
        cutout_transform = Cutout(n_holes=n_holes, length=16)
        return cutout_transform(img)

    def get_n_holes_dict(self):
        return self.n_holes_dict
    

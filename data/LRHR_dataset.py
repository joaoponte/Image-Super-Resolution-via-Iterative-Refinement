from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util

import matplotlib.pyplot as plt

class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split

        if datatype == 'lmdb':
            self.env = lmdb.open(dataroot, readonly=True, lock=False,
                                 readahead=False, meminit=False)
            # init the datalen
            with self.env.begin(write=False) as txn:
                self.dataset_len = int(txn.get("length".encode("utf-8")))
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype == 'img':
            self.sr_path = Util.get_paths_from_images(
                '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
            self.hr_path = Util.get_paths_from_images(
                '{}/hr_{}'.format(dataroot, r_resolution))
            if self.need_LR:
                self.lr_path = Util.get_paths_from_images(
                    '{}/lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        if self.datatype == 'lmdb':
            with self.env.begin(write=False) as txn:
                hr_img_bytes = txn.get(
                    'hr_{}_{}'.format(
                        self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                sr_img_bytes = txn.get(
                    'sr_{}_{}_{}'.format(
                        self.l_res, self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                if self.need_LR:
                    lr_img_bytes = txn.get(
                        'lr_{}_{}'.format(
                            self.l_res, str(index).zfill(5)).encode('utf-8')
                    )
                # skip the invalid index
                while (hr_img_bytes is None) or (sr_img_bytes is None):
                    new_index = random.randint(0, self.data_len-1)
                    hr_img_bytes = txn.get(
                        'hr_{}_{}'.format(
                            self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    sr_img_bytes = txn.get(
                        'sr_{}_{}_{}'.format(
                            self.l_res, self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    if self.need_LR:
                        lr_img_bytes = txn.get(
                            'lr_{}_{}'.format(
                                self.l_res, str(new_index).zfill(5)).encode('utf-8')
                        )
                img_HR = Image.open(BytesIO(hr_img_bytes)).convert("RGB")
                img_SR = Image.open(BytesIO(sr_img_bytes)).convert("RGB")
                if self.need_LR:
                    img_LR = Image.open(BytesIO(lr_img_bytes)).convert("RGB")
        else:
            img_HR = Image.open(self.hr_path[index]).convert("RGB")
            img_SR = Image.open(self.sr_path[index]).convert("RGB")
            if self.need_LR:
                img_LR = Image.open(self.lr_path[index]).convert("RGB")
        if self.need_LR:
            [img_LR, img_SR, img_HR] = Util.transform_augment(
                [img_LR, img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index}
        else:
            [img_SR, img_HR] = Util.transform_augment(
                [img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'HR': img_HR, 'SR': img_SR, 'Index': index}

class LRHRDatasetJIT(Dataset):
    def __init__(self, data_lr, data_hr, coordinates_lr, value_range_lr, value_range_hr, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False, rgb=True):
        self.datatype = datatype
        self.l_resolution = l_resolution
        self.value_range_lr = value_range_lr
        self.r_resolution = r_resolution
        self.value_range_hr = value_range_hr
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split
        self.coords = coordinates_lr
        self.rgb = rgb
        print('rgb', self.rgb)
        

        if datatype=='numpy':
            self.dataset_len = data_lr.shape[0]

            self.data_lr = data_lr
            self.data_hr = data_hr

            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        if self.datatype == 'numpy':
            # print('img original HR', self.data_hr[index].min(), self.data_hr[index].max(), self.value_range_hr)
            img_HR = Util.transform2PIL(self.data_hr[index], self.rgb, self.value_range_hr)
            # print('img PIL HR', *img_HR.getextrema(), self.value_range_hr)

            if self.need_LR:
                # print('img original LR', self.data_lr[index].min(), self.data_lr[index].max(), self.value_range_lr)
                img_SR = Util.transform2PIL(self.data_lr[index], self.rgb, self.value_range_lr)
                # print('img PIL HR', *img_SR.getextrema(), self.value_range_lr)

                img_LR = Util.transform2PIL(self.data_lr[index], self.rgb, self.value_range_lr)
                img_LR = img_LR.resize((self.l_resolution,self.l_resolution), resample=Image.BICUBIC)
                # print('img PIL HR', *img_LR.getextrema(), self.value_range_lr)
            else:
                img_SR = Util.transform2PIL(self.data_hr[index], self.rgb, self.value_range_hr)
                img_SR = img_SR.resize((self.l_resolution,self.l_resolution), resample=Image.BICUBIC)
                img_SR = img_SR.resize((self.r_resolution,self.r_resolution), resample=Image.BICUBIC)

        if self.need_LR:
            [img_LR, img_SR, img_HR] = Util.transform_augment(
                [img_LR, img_SR, img_HR], 
                img_value_ranges=[self.value_range_lr, self.value_range_lr, self.value_range_hr], 
                split=self.split, 
                min_max=(-1, 1)
            )
            return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index, 'Coords': self.coords[index]}
        else:
            [img_SR, img_HR] = Util.transform_augment(
                [img_SR, img_HR], 
                img_value_ranges=[self.value_range_lr, self.value_range_hr], 
                split=self.split, 
                min_max=(-1, 1)
            )
            return {'HR': img_HR, 'SR': img_SR, 'Index': index, 'Coords': self.coords[index]}
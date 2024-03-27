import os
import cv2
import json
import numpy as np
import PIL.Image as Image
import SimpleITK as sitk
import torch
import torch.nn as nn 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import skimage.measure as measure

from preprocess import crop_pad 
from residual_unet import ResidualUNet 


def compress(p_list, compress_degree=20):
    written_list = []
    c = 1e-3
    for i in range(0, p_list.shape[0]):
        ele = p_list[i, :]
        if i > 0 and i < p_list.shape[0] - 1:
            first_dir = ele - p_list[i - 1, :]  # 向量1数值
            second_dir = p_list[i + 1, :] - ele  # 向量2数值
            vector = first_dir * second_dir
            v = np.sqrt((vector * vector).sum())
            ab = v / (
                    np.sqrt((first_dir * first_dir).sum())
                    * np.sqrt((second_dir * second_dir).sum())
                )
            if ab >= 1 - c and ab <= 1 + c:
                continue
            elif ab >= -1 - c and ab <= -1 + c:
                continue
            elif ab >= 0 - c and ab <= 0 + c:
                continue
            last_p = written_list[len(written_list) - 1]
            dis = ele - last_p
            if np.sqrt((dis * dis).sum()) < compress_degree * np.sqrt(2):
                continue
        written_list.append(ele)
    ret = np.array(written_list)
    return ret


def approx_contour(base_contour, new_contour, alpha):
    # base_contour.shape = (N1, 1, 2)
    # new_contour.shape = (N2, 1, 2)
    new_contour = new_contour.reshape(1, -1, 2)
    # new_contour.shape = (1, N2, 2)
    dist = np.sqrt(np.sum(np.square(base_contour - new_contour), axis=2))
    # dist.shape = (N1, N2)
    match_indexes = np.argmin(dist, axis=1)
    match_contour = new_contour[0][match_indexes].reshape(-1, 1, 2)
    res_contour = (1 - alpha) * base_contour + alpha * match_contour
    # res_contour = np.round(res_contour).astype(np.int)
    print(base_contour.shape, res_contour.shape)
    return res_contour


def get_contour(mask, compress_degree):
    # Get contour
    h = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = h[0]
    contours.sort(key=lambda x : cv2.contourArea(x), reverse=True)
    # contours = [cv2.convexHull(contour) for contour in contours]
    # contour = cv2.convexHull(contours[0]).astype(np.float32)
    contour = contours[0].astype(np.float32)
    print('Contour area is', cv2.contourArea(contour))
    contour = compress(contour, compress_degree)
    return contour


def clip_value(image):
    min_value = np.percentile(image, 1)
    max_value = np.percentile(image, 99)
    return np.clip(image, min_value, max_value)


def normalize(img, range_norm=False):
    if range_norm:
        mn = img.min()
        mx = img.max()
        img = (img - mn) / (mx - mn)
    mean = img.mean()
    std = img.std()
    denominator = np.reciprocal(std)
    img = (img - mean) * denominator
    return img


def read_image(image_path, united_shape=[-1, 288, 288], pre_crop_type=[-4, -3, -2]):
    image_itk = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(image_itk)
    ori_shape = image.shape
    image = crop_pad(image, pre_crop_type, united_shape)
    image = clip_value(image)
    image = normalize(image)
    return torch.tensor(image).unsqueeze(0).float(), ori_shape, image_itk


def restore(pred, ori_shape, pre_crop_type=[-4, -3, -2]):
    pred = crop_pad(pred, pre_crop_type, ori_shape)
    return pred


class Annotator():

    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model = ResidualUNet(num_classes=12, infer_size=96).to(self.device)
        ckpt = torch.load('ckpts/ResidualUNet_epoch_79.pth')
        state_dict = ckpt['state_dict']
        state_dict = dict([(key.replace('module.', ''), value) for key, value in state_dict.items()])
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.cache = {}

    def predict(self, collection_name, image_name, compress_degree):
        if collection_name in self.cache.keys():
            seg_res = self.cache[collection_name]
            print('Load result from cache')
        elif os.path.exists(f'sitk_images/{collection_name}_pred.nii.gz'):
            seg_res = sitk.GetArrayFromImage(sitk.ReadImage(f'sitk_images/{collection_name}_pred.nii.gz'))
            self.cache[collection_name] = seg_res
            print('Load previous result')
        else:
            image, ori_shape, image_itk = read_image(f'sitk_images/{collection_name}.nrrd')
            image = image.unsqueeze(0).to(self.device)
            with torch.no_grad():
                seg_score, seg_res = self.model(image)
                seg_score = seg_score.squeeze(0).cpu().numpy()
                seg_res = seg_res.squeeze(0).cpu().numpy()
            seg_res = restore(seg_res, ori_shape)
            seg_res_itk = sitk.GetImageFromArray(seg_res)
            seg_res_itk.CopyInformation(image_itk)
            sitk.WriteImage(seg_res_itk, f'sitk_images/{collection_name}_pred.nii.gz')
            self.cache[collection_name] = seg_res
            print('Predict now')
            # for slice_ind in range(image.shape[0]):
            #     json_path = f'json/{collection_name}/{slice_ind:04}.json'
            #     if not os.path.exists(json_path):
            #         with open(json_path, 'w') as writer:
            #             writer.write('[]')
            #     with open(json_path, 'r') as reader:
            #         content = eval(reader.read())
            #     if len(content) == 0:
            #         pred = seg_res[slice_ind]
            #         # TODO
            #         with open(json_path, 'w') as writer:
            #             json.dumps(anno, indent=4)
            # print('Fill results')
        pred = seg_res[int(image_name.replace('.jpg', ''))]
        unique_labels = np.unique(pred)[1: ]  # ignore background
        res = {}
        for label_id in unique_labels:
            labels = measure.label(pred == label_id, connectivity=2, background=0)
            regions = measure.regionprops(labels)
            max_idx = np.argmax(np.bincount(labels.reshape(-1))[1 :]) + 1
            mask = (labels == max_idx).astype(np.uint8)
            contour = get_contour(mask, compress_degree)
            res[str(label_id)] = [
                {'x': int(point[0, 0]), 'y': int(point[0, 1])}
                for point in contour
            ]
        return res


def test_contour():
    base_contour = np.array([[0, 0], [0, 1], [1, 1], [1, 0]]).reshape(-1, 1, 2)
    new_contour = np.array([[-1, 0], [0, 2], [2, 0]]).reshape(-1, 1, 2)
    res = approx_contour(base_contour, new_contour, 0.5)
    print(res.shape)
    print(res)


if __name__ == '__main__':
    annotator = Annotator()
    collect_name = '1_40'
    image_name = '0030.jpg'
    res = annotator.predict(collect_name, image_name, 0)
    image = cv2.imread(f'images/{collect_name}/{image_name}')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for label_id in res:
        start_point = res[label_id][0]
        prev_point = start_point
        for point in res[label_id][1 :]:
            cv2.line(image, (prev_point['x'], prev_point['y']), (point['x'], point['y']), [255, 0, 0])
            prev_point = point
        cv2.line(image, (prev_point['x'], prev_point['y']), (start_point['x'], start_point['y']), [255, 0, 0])
    cv2.imwrite('test_anno.png', image)

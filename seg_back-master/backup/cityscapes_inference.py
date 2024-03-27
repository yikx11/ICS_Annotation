import torch
import numpy as np
import os
import PIL.Image as Image


def get_next_offset(x, x_max, size, stride):
    if x == x_max - size:
        return -1
    x += stride
    if x <= x_max - size:
        return x
    elif x - stride + size != x_max:
        return x_max - size
    else:
        return -1


def patch_predict(model, images, cfg):
    batch_size = images.size(0)
    h = images.size(2)
    w = images.size(3)
    res = torch.zeros(batch_size, 19, h, w).to(cfg.device)
    y = 0
    while y >= 0:
        x = 0
        while x >= 0:
            patch_batch = images[
                :, :, y : y + cfg.input_size, x : x + cfg.input_size
            ]
            with torch.no_grad():
                output = model(patch_batch)
            res[:, :, y : y + cfg.input_size, x : x + cfg.input_size] = output
            x = get_next_offset(x, w, cfg.input_size, cfg.infer_stride)
        y = get_next_offset(y, h, cfg.input_size, cfg.infer_stride)
    return res


def inference(model, images):
    # with torch.no_grad():
    #     output = model(images)
    output = patch_predict(model, images, cfg)
    preds = torch.max(output, dim=1)[1]
    preds = preds.cpu().data.numpy().astype(np.uint8)
    for index, pred in zip(indexes, preds):
        image_name = image_paths[index].split('/')[-1]
        Image.fromarray(pred).save(save_path + '/' + image_name)
    return output
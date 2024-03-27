import os
import SimpleITK as sitk
import numpy as np
import PIL.Image as Image


def to_uint8(image):
    min_value = np.percentile(image, 1)
    max_value = np.percentile(image, 99)
    image = (np.clip(image, min_value, max_value) - min_value) / (max_value - min_value)
    uint8_image = np.round(image * 127).astype(np.uint8)
    return uint8_image


def main():
    src_dir = 'sitk_images'
    dst_dir = 'images'
    sitk_image_names = os.listdir(src_dir)
    for sitk_image_name in sitk_image_names:
        key = sitk_image_name.replace('.nrrd', '').replace('.nii.gz', '')
        if not os.path.exists(dst_dir + '/' + key):
            os.mkdir(dst_dir + '/' + key)
        sitk_image = sitk.GetArrayFromImage(sitk.ReadImage(src_dir + '/' + sitk_image_name))
        uint8_images = to_uint8(sitk_image)
        for idx, image in enumerate(uint8_images):
            Image.fromarray(image).save(f'{dst_dir}/{key}/{idx:04}.jpg')


if __name__ == '__main__':
    main()

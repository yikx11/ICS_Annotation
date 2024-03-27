import os
import argparse
import json


def main():
    dir_names = os.listdir('images')
    for dir_name in dir_names:
        image_names = os.listdir('images/' + dir_name)
        image_names.sort()
        res = []
        for image_name in image_names:
            flag = True
            for key in ['_0.25', '_0.5', '_1.0', '_2.0', '_4.0']:
                if key in image_name:
                    flag = False
                    break
            if not flag:
                continue
            if image_name.endswith('.jpg') or image_name.endswith('.png'):
                res.append({'image_name': image_name})
        with open(f'info/{dir_name}.json', 'w') as writer:
            writer.write(json.dumps(res, indent=4))
        if not os.path.exists('json/' + dir_name):
            os.makedirs('json/' + dir_name)


if __name__ == '__main__':
    main()

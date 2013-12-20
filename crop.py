import Image
import sys
import os

STANDARD_SIZE = (100, 100)

def crop_car(image_path, image_name, target_path, debug = False):
    img = Image.open(image_path)
    if debug:
        img.show()
    left = 170
    top = 110
    width = 300
    height = 300
    box = (left, top, left+width, top+height)
    area = img.crop(box)
    area = area.resize(STANDARD_SIZE)
    if debug:
        area.show()
    else:
        area.save(target_path + image_name)


dir = './taken/'

for img in os.listdir(dir):
    print img
    crop_car(dir + img, img, './taken-crop/')

dir = './available/'

for img in os.listdir(dir):
    print img
    crop_car(dir + img, img, './available-crop/')

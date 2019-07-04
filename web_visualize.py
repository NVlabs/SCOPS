"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from web_vis_html import HTML
import os, argparse
from shutil import copyfile

parser = argparse.ArgumentParser(description='visualize images in different folders')
parser.add_argument('-dirs',  nargs='+', help='dirs')
parser.add_argument('-ref', type=int, default=3, help='reference dir index')
parser.add_argument('-names', nargs='+', help='names')
parser.add_argument('-o', default='', help='output path')


args = parser.parse_args()



# take dir1 as reference
imgs = [img[:-4] for img in os.listdir(args.dirs[args.ref]) if (img[-3:] == 'jpg' or img[-3:] == 'png')]
print('total {} images'.format(len(imgs)))

webpage = HTML(args.o, 'visualize')
webpage.add_header('image comparisons')


img_type = []
for d in args.dirs:
	sub_imgs = [img for img in os.listdir(d) if (img[-3:] == 'jpg' or img[-3:] == 'png')]
	img_type.append(sub_imgs[0][-4:])

for name in args.names:
	if not os.path.exists(os.path.join(args.o,'images',name)):
		os.makedirs(os.path.join(args.o,'images',name))

for img in imgs:
	links = []
	for c in range(len(args.dirs)):
		links.append(os.path.join(args.names[c], img+img_type[c]))
		copyfile(os.path.join(args.dirs[c], img+img_type[c]), os.path.join(args.o, 'images', args.names[c], img+img_type[c]))
	webpage.add_images(links, args.names, links)

webpage.save()

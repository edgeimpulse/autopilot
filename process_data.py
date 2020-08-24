from PIL import Image, ImageOps
import os, sys
from os import path

if not path.exists('tmp'):
    os.mkdir('tmp')
if not path.exists('jpg'):
    os.mkdir('jpg')

filename = sys.argv[2].split('/')[2].split('.')[0]

os.system('rm -rf tmp/*')
os.system('rm -rf jpg/*')
os.mkdir('tmp/' + filename)
os.mkdir('jpg/' + filename)

os.system('convert -coalesce ' + sys.argv[2] + ' ' + 'jpg/' + filename + '/' + filename + '.jpg')

with open(sys.argv[2].replace('.gif', '.txt')) as f:
    lines = [line.rstrip() for line in f]
    for ix in range(len(lines)):
        angle = int(float(lines[ix]))

        im = Image.open("jpg/" + filename + '/' + filename + '-' + str(ix) + '.jpg')
        im.save('tmp/' + filename + '/' + str(angle) + '.' + filename + '-' + str(ix) + '.jpg', quality=100)

        im_mirror = ImageOps.mirror(im)
        im_mirror.save('tmp/' + filename + '/' + str(-angle) + '.' + filename + '-' + str(ix) + '-m' + '.jpg', quality=100)

os.system('edge-impulse-uploader --allow-duplicates --category ' + sys.argv[1] + ' tmp/' + filename + '/*')
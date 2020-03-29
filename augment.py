import argparse

from torchvision import datasets
import torchvision.transforms.functional as TFunc

parser = argparse.ArgumentParser(description='Create augmented versions of images. Images should be in ImageFolder structure.')
parser.add_argument('folder', type=str, help='Folder location')
args = parser.parse_args()
data = datasets.ImageFolder(args.folder)

i = 0

print(f'Augmenting {len(data.imgs)} images...')

# horizontal flips
for img in data:
    # save each image in the folder we found it in with the name a0.png, a1.png, ...
    TFunc.hflip(img[0]).save(f"{args.folder}/{data.classes[img[1]]}/a{i}.png", "png")
    i += 1

print('Done with horizontal flips')

# vertical flips
for img in data:
    TFunc.vflip(img[0]).save(f"{args.folder}/{data.classes[img[1]]}/a{i}.png", "png")
    i += 1

print('Done with vertical flips')

# rotate 90 degrees
for img in data:
    TFunc.rotate(img[0], 90).save(f"{args.folder}/{data.classes[img[1]]}/a{i}.png", "png")
    i += 1

print('Done with 90 degree rotation')

# rotate 180 degrees
for img in data:
    TFunc.rotate(img[0], 180).save(f"{args.folder}/{data.classes[img[1]]}/a{i}.png", "png")
    i += 1

print('Done with 180 degree rotation')

# rotate 270 degrees
for img in data:
    TFunc.rotate(img[0], 270).save(f"{args.folder}/{data.classes[img[1]]}/a{i}.png", "png")
    i += 1

exit(0)




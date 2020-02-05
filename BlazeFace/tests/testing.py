import torch
import torch.nn as nn
from PIL import Image
from ssd import build_ssd
#from data import VOC_ROOT, VOC_CLASSES as labelmap

def test_net(save_folder, net, cuda, testset, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder+'test1.txt'
    print(filename)
    num_images = len(testset)
    path = os.getcwd() + '/data/WIDER/WIDER_test/images'

    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.jpg' in file:
                files.append(os.path.join(r, file))


    for i in range(num_images):
        # print(i)
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img = testset.pull_image(files[i])
        print(testset._imgpath)
        img_id, annotation = testset.pull_anno(files[i])
        # print(img)
        print(img_id)

        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))


        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        pred_num = 0
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.4:
                if pred_num == 0:
                    with open(filename, mode='a') as fi:
                        print("here------------------")
                        fi.write('PREDICTIONS: '+ img_id + '\n')
                score = detections[0, i, j, 0]
                label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                #append to a text file
                pred_num += 1
                with open(filename, mode='a') as fi:
                    fi.write(str(pred_num)+' label: '+label_name+' score: ' +
                            str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
                j += 1


def test():
    num_classes = len(VOC_CLASSES) + 1
    net = build_blazeface('test', 300, num_classes)
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    testset = WIDERDetection(args.voc_root, ['wider_test'], None, WIDERAnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation

    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold)

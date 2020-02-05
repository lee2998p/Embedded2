import argparse
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
import sys
import time
import warnings
import cv2
import torch
from data import BaseTransform
from torch.autograd import Variable
import atexit
from ssd import build_ssd

SIZE_THRESH = 600
warnings.filterwarnings("once")


def loop_test(network, device, transformer, img_q: Queue, bbox_q: Queue, threshold=0.35):
    scale = None
    print(f"NETWORK IS NONE {type(network)}")
    print("STARTING TO SPIN DETECT LOOP")
    while True:
        print("WAIT")
        image = img_q.get()
        print("RECV")
        if type(image) is str and image == "DONE":
            del image
            break
        print("CHECK")
        boxes = detect_face(image, network, transformer, device, threshold)

        print("SENDING")
        bbox_q.put(boxes)
        print("SENT")
        # DONT FORGET TO CLEANUP
        del image
    img_q.close()
    bbox_q.close()
    print("BYE")


def detect_face(image, network, tformer, device, threshold=0.35):
    print("Start detect")
    x = torch.from_numpy(tformer(image)[0]).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0)).to(device)
    y = network(x)
    print("DETECTION")

    detections = y.data

    scale = torch.Tensor([image.shape[1], image.shape[0],
                          image.shape[1], image.shape[0]])

    boxes = []
    j = 0
    while detections[0, 1, j, 0] >= threshold:
        pt = (detections[0, 1, j, 1:] * scale).cpu().numpy()
        x1, y1, x2, y2 = pt
        if x2 - x1 < SIZE_THRESH and y2 - y1 < SIZE_THRESH:
            boxes.append((x1, y1, x2, y2))
        j += 1
    return boxes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single Shot MultiBox Detection")
    parser.add_argument('--trained_model', default='ssd300_WIDER_100455.pth', type=str, help="Trained state_dict file")
    parser.add_argument('--visual_threshold', default=0.6, type=float, help='Final confidence threshold')
    parser.add_argument('--cuda', default=False, type=bool)
    parser.add_argument('--encrypt', default=False, type=bool, help='Enable/Disable encryption')
    parser.add_argument('--camera_index', default=0, type=int, help='Index of camera')
    parser.add_argument('--drop_rate', default=15, type=int, help='Take 1 out of the drop rate and process')
    parser.add_argument('--window_name', default="Face Detection -- Frame Drop", type=str,
                        help="Name for the display window")
    args = parser.parse_args()

    if args.cuda and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda:0')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
        device = torch.device('cpu')

    cap = cv2.VideoCapture(args.camera_index)

    if not cap.isOpened():
        print("Camera failed to open")
        sys.exit(1)
    cv2.namedWindow(args.window_name, cv2.WINDOW_AUTOSIZE)

    num_classes = 2
    net = build_ssd('test', 300, num_classes)
    net.load_state_dict(torch.load(args.trained_model, map_location=device))
    net.eval()
    print('Model has been loaded')

    transformer = BaseTransform(net.size, (104, 117, 123))

    encrypt_status = 1
    decrypt_status = 1
    verbose = 0
    frame_count = 0

    mp.set_start_method('forkserver')
    img_q = Queue()
    box_q = Queue()
    worker_proc = Process(target=loop_test, args=(net, device, transformer, img_q, box_q,))
    worker_proc.start()
    boxes = []
    wait_update = False


    @atexit.register
    def cleanup():
        img_q.put("DONE")
        worker_proc.terminate()
        cv2.destroyAllWindows()
        print("BYE BYE")


    while True:
        start = time.time()
        ret, image = cap.read()
        image = cv2.flip(image, 1)
        if frame_count % args.drop_rate == 0 or wait_update:
            print("TRY SEND")
            img_q.put(image)
            print("SENT")
            # Cleanup the old boxes
            del boxes
            boxes = box_q.get()
            wait_update = False
            # boxes = detect_face(image, net, device)

        for box in boxes:
            x1, y1, x2, y2 = box
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        end = time.time()
        fps = 1 / (end - start)
        image = cv2.putText(image, 'fps: %.3f' % fps, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv2.imshow(args.window_name, image)
        key = cv2.waitKey(1)
        if key == 27:
            break

        frame_count += 1

    img_q.put("DONE")

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from blazeface import BlazeFace


device = torch.device("cpu")


def plot_detections(img, detections, with_keypoints=True):
    #fig, ax = plt.subplots(1, figsize=(10, 10))
    #ax.grid(False)
    #ax.imshow(img)

    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()

    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)

    print("Found %d faces" % detections.shape[0])

    for i in range(detections.shape[0]):
        ymin = detections[i, 0] * img.shape[0]
        xmin = detections[i, 1] * img.shape[1]
        ymax = detections[i, 2] * img.shape[0]
        xmax = detections[i, 3] * img.shape[1]

        #rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
        #                         linewidth=1, edgecolor="r", facecolor="none",
        #                         alpha=detections[i, 16])
        #ax.add_patch(rect)
        print(xmax, xmin)
        img = img/255.0

        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255))

        if with_keypoints:
            for k in range(6):
                kp_x = detections[i, 4 + k * 2] * img.shape[1]
                kp_y = detections[i, 4 + k * 2 + 1] * img.shape[0]
                #circle = patches.Circle((kp_x, kp_y), radius=0.5, linewidth=1,
                #                        edgecolor="lightskyblue", facecolor="none",
                #                        alpha=detections[i, 16])
                #ax.add_patch(circle)
                cv2.circle(img, (int(kp_x), int(kp_y)), 2, (0, 255, 0))
    return img

    #plt.show()

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    net = BlazeFace().to(device)
    net.load_weights("blazeface.pth")
    net.load_anchors("anchors.npy")

    # Optionally change the thresholds:
    net.min_score_thresh = 0.75
    net.min_suppression_threshold = 0.3

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (128, 128)).astype(np.float32)

        print(frame.shape)

        detections = net.predict_on_image(frame)
        img = plot_detections(frame, detections)
        print(detections.shape)
        # #detections = detections.squeeze()
        # top_left = tuple((detections[0][0], detections[0][1]))
        # bottom_right = tuple((detections[0][1], detections[0][3]))
        # frame = frame/255.0

    #     cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255))
    #     cv2.imshow('frame', frame)
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("output", 600, 600)
        cv2.imshow('output', img)

        cv2.waitKey(1)

    cv2.destroyAllWindows()
    exit()


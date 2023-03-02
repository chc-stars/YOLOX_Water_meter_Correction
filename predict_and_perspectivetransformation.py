# -------------------------------
# -*- coding = utf-8 -*-
# @Time : 2022/7/26 18:58
# @Author : chc_stars
# @File : predict_and_perspectivetransformation.py
# @Software : PyCharm
# -------------------------------

import os
import cv2
import numpy as np
from models.yolox import Detector
from utils.util import mkdir, label_color, get_img_path
from config import opt
import tqdm
import time


def vis_result(img, results):
    for res_i, res in enumerate(results):
        label, conf, bbox = res[:3]
        bbox = [int(i) for i in bbox]
        if len(res) > 3:
            reid_feat = res[4]
            print("reid feat dim {}".format(len(reid_feat)))

        color = label_color[opt.label_name.index(label)]
        # show box
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        # show label and conf
        txt = '{}:{:.2f}'.format(label, conf)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
        cv2.rectangle(img, (bbox[0], bbox[1] - txt_size[1] - 2), (bbox[0] + txt_size[0], bbox[1] - 2), color, -1)
        cv2.putText(img, txt, (bbox[0], bbox[1] - 2), font, 0.5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    return img


def detect():
    # img_dir = "E:\Experimental_records\\water_meter\\qingxie"
    img_dir = "ces"
    output = "qinxie"
    mkdir(output, rm=True)

    img_list = get_img_path(img_dir, extend='.jpg')
    assert len(img_list) != 0, "cannot find img in {}".format(img_dir)

    detector = Detector(opt)

    for index, image_path in enumerate(tqdm.tqdm(img_list)):
        print("------------------------------")
        print("{}/{}, {}".format(index, len(img_list), image_path))


        assert os.path.isfile(image_path), "cannot find {}".format(image_path)
        img = cv2.imread(image_path)
        s1 = time.time()
        results = detector.run(img, vis_thresh=opt.vis_thresh, show_time=True)
        print("[pre_process + inference + post_process] time cost: {}s".format(time.time() - s1))
        print(results)

        if len(results) > 0:
            for i in range(len(results)):
                if "Water_Meter_Dial" not in results[i]:
                    continue
                else:
                    results = sorted(results)
                    water_dial = results[-1:]
                    print(water_dial)
                    water_dial = water_dial[0][2]
                    print(water_dial)
                    h, w = img.shape[: 2]


                    # cv2.imshow('s', img)
                    # cv2.waitKey(0)

                    src = np.float32([[water_dial[0], water_dial[1]], [water_dial[2], water_dial[1]],
                                      [water_dial[0], water_dial[3]], [water_dial[2], water_dial[3]]])
                    dst = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

                    M = cv2.getPerspectiveTransform(src, dst)

                    if h > w:

                        new = cv2.warpPerspective(img, M, (h, h))
                    elif h < w:
                        new = cv2.warpPerspective(img, M, (w, w))
                    else:
                        new = cv2.warpPerspective(img, M, (h, h))

                    ts_picture_path = output + "/" + image_path.split("/")[-2] + "/" + "ts_picture"
                    mkdir(ts_picture_path)
                    ts_picture = ts_picture_path + '/' + os.path.basename(image_path)
                    cv2.imwrite(ts_picture, new)
                    print("save the picture as {}".format(ts_picture_path))

        # ----------------------------------------------------
        img = vis_result(img, results)
        save_p = output + "/" + image_path.split("/")[-2]
        mkdir(save_p)
        save_img = save_p + "/" + os.path.basename(image_path)
        cv2.imwrite(save_img, img)
        print("save image to {}".format(save_img))


if __name__ == '__main__':
    opt.load_model = opt.load_model if opt.load_model != "" else os.path.join(opt.save_dir, "model_best.pth")

    detect()
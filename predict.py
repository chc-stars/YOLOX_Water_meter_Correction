
import os
import cv2
import tqdm
import time

from config import opt
from models.yolox import Detector
from utils.util import mkdir, label_color, get_img_path

from angle_cla import get_angle, get_angle_waterdial_readarea

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

    # img_dir = opt.dataset_path + "/images/val2017" if "img_dir" not in opt else opt["img_dir"]
    # img_dir = "F:\water-meter-data\jiaozheng\img_normal_train_2"
    # img_dir = "F:\water-meter-data\jiaozheng_jiucuo"
    img_dir = "F:\\water-meter-data\\2204011649205354769_jiaozheng"
    is_save_det_pic = True  # 是否保存检测后的图像

    output = "output_newsd"
    mkdir(output, rm=True)

    img_list = get_img_path(img_dir, extend=".jpg")
    assert len(img_list) != 0, "cannot find img in {}".format(img_dir)


    # 矫正路径
    Corrective_picture_path = output + "/" + img_list[0].split("/")[-2] + "/" + "Corrective_picture"
    mkdir(Corrective_picture_path)
    # 未矫正路径
    Uncorrected_picture_path = output + "/" + img_list[0].split("/")[-2] + "/" + "Uncorrected_picture"
    mkdir(Uncorrected_picture_path)

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

        # __________________________________________
        # 对识别物体进行判断：存在以下情况不进行矫正：（默认情况下一张图片只有一个水表盘）
        #     1 、 未检测到读数区域，（检测到表盘（是或否）， 检测到单位（是或否））
        #     2 、 检测到读数区域，（未检测到单位， 检测到表盘（检测表盘是否完整（否））
        # 以下情况进行矫正 （默认情况下一张图片只有一个水表盘）：
        #     1、  检测到读数区域，检测到单位 , 检测到表盘  （3）
        #     2、  检测到读数区域， 未检测到单位，  检测到完整表盘  （2）
        #     3、  检测到读数区域， 检测到单位， 未检测到表盘  （2）

        # ----------------------------------------------
        #     对未校正的图片单独存放在文件夹

        if len(results) > 1 and len(results) <= 3:

            if len(results) == 3:
                results = sorted(results)
                if "Reading_Area" in results[0] and "Unit" in results[1] and "Water_Meter_Dial" in results[2]:
                    boxes_center = []
                    for i in range(3):

                        box_ = results[i][2]
                        box_center = [(box_[2] - box_[0]) / 2 + box_[0], (box_[3] - box_[1]) / 2 + box_[1]]
                        boxes_center.append(box_center)

                    # boxes[0] 是读数区中点， boxes[1] 是单位中点， boxes[2] 是表盘中点
                    boxes_center_last = boxes_center[:2]
                    point_read_area_unit_center = [boxes_center[1][0], boxes_center[0][1]]
                    boxes_center_last.append(point_read_area_unit_center)

                    angle_ = get_angle(boxes_center_last[0], boxes_center_last[1], boxes_center_last[2])

                    rows, cols = img.shape[: 2]

                    # 仿射变换
                    M = cv2.getRotationMatrix2D(center=boxes_center_last[0], angle=angle_, scale=1)
                    res = cv2.warpAffine(img, M, (cols, rows))

                    Corrective_picture = Corrective_picture_path + '/' + os.path.basename(image_path)
                    cv2.imwrite(Corrective_picture, res)
                    print("save image to {}".format(Corrective_picture))

                else:
                    Uncorrected_picture = Uncorrected_picture_path + '/' + os.path.basename(image_path)
                    cv2.imwrite(Uncorrected_picture, img)

            else:   #  len(results) ==2
                results = sorted(results)
                results_first = results[0]
                results_last = results[1]

                if "Reading_Area" not in results_first and "Reading_Area" not in results_last:
                    Uncorrected_picture = Uncorrected_picture_path + '/' + os.path.basename(image_path)
                    cv2.imwrite(Uncorrected_picture, img)
                    print("If the correction conditions are not met, save the picture as {}".format(Uncorrected_picture_path))

                elif "Reading_Area" in results_first and "Reading_Area" in results_last:
                    Uncorrected_picture = Uncorrected_picture_path + '/' + os.path.basename(image_path)
                    cv2.imwrite(Uncorrected_picture, img)
                    print("If the correction conditions are not met, save the picture as {}".format(
                        Uncorrected_picture_path))
                elif "Reading_Area" in results_first and "Unit" in results_last:
                    results_first_center = [(results_first[2][2] - results_first[2][0]) / 2 + results_first[2][0], (results_first[2][3] - results_first[2][1]) / 2 + results_first[2][1]]
                    results_last_center = [(results_last[2][2] - results_last[2][0]) / 2 + results_last[2][0], (results_last[2][3] - results_last[2][1]) / 2 + results_last[2][1]]
                    results_first_and_last_center = [results_last_center[0], results_first_center[1]]
                    angle_ = get_angle(results_first_center, results_last_center, results_first_and_last_center)
                    rows, cols = img.shape[: 2]

                    # 仿射变换
                    M = cv2.getRotationMatrix2D(center=results_first_center, angle=angle_, scale=1)

                    res = cv2.warpAffine(img, M, (cols, rows))

                    Corrective_picture = Corrective_picture_path + '/' + os.path.basename(image_path)
                    cv2.imwrite(Corrective_picture, res)
                    print("save image to {}".format(Corrective_picture))

                    del results_first, results_last

                elif "Reading_Area" in results_first and "Water_Meter_Dial" in results_last:
                    # 判读表盘是否完整
                    threshold = 0.05

                    h, w = img.shape[: 2]

                    if results_first[2][0] / w < threshold or results_first[2][1] / h < threshold or results_first[2][2] / w > (1 - threshold) or results_first[2][3] / h > (1 - threshold):
                        print("表盘显示不完整，不做矫正")
                        Uncorrected_picture = Uncorrected_picture_path + '/' + os.path.basename(image_path)
                        cv2.imwrite(Uncorrected_picture, img)


                    else:

                        results_first_center = [(results_first[2][2] - results_first[2][0]) / 2 + results_first[2][0],
                                        (results_first[2][3] - results_first[2][1]) / 2 + results_first[2][1]]  # 读数中心
                        results_last_center = [(results_last[2][2] - results_last[2][0]) / 2 + results_last[2][0],
                                       (results_last[2][3] - results_last[2][1]) / 2 + results_last[2][1]]  # 表盘中心
                        results_first_and_last_center = [results_last_center[0], results_first_center[1]]

                        angle_ = get_angle_waterdial_readarea(results_last_center, results_first_center, results_first_and_last_center)

                        rows, cols = img.shape[: 2]

                        # 仿射变换
                        M = cv2.getRotationMatrix2D(center=results_first_center, angle=angle_, scale=1)

                        res = cv2.warpAffine(img, M, (cols, rows))

                        Corrective_picture = Corrective_picture_path + '/' + os.path.basename(image_path)
                        cv2.imwrite(Corrective_picture, res)
                        print("save image to {}".format(Corrective_picture))
                else:

                    Uncorrected_picture = Uncorrected_picture_path + '/' + os.path.basename(image_path)
                    cv2.imwrite(Uncorrected_picture, img)
        else:

            Uncorrected_picture = Uncorrected_picture_path + '/' + os.path.basename(image_path)
            cv2.imwrite(Uncorrected_picture, img)
            print("If the correction conditions are not met, save the picture as {}".format(Uncorrected_picture_path))

        # ------------------------------------------

        if is_save_det_pic:
            img = vis_result(img, results)
            save_p = output + "/" + image_path.split("/")[-2]
            mkdir(save_p)
            save_img = save_p + "/" + os.path.basename(image_path)
            cv2.imwrite(save_img, img)
            print("save image to {}".format(save_img))


if __name__ == "__main__":
    opt.load_model = opt.load_model if opt.load_model != "" else os.path.join(opt.save_dir, "model_best.pth")
    detect()


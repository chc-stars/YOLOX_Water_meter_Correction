

import math
from PIL import Image

def get_angle(point_reading_area_center, point_unit_center, point_read_area_unit_center):

    """             (line_prauc_prac)
    (prac)* -----------------------------*(prauc)
           |                             |
           |                             |
           |                             |(line_puc_prauc)
           |                             |
           |                             |
           | ----------------------------*(puc)
    Args:
        point_reading_area_center (prac): 读数区中点 [x, y]
        point_unit_center (puc): 单位中点 [x, y]
        point_read_area_unit_center (prauc): 与读数区水平，与单位区域垂直 [x, y]

    Returns: 矫正角度

    """

    line_prauc_prac = point_read_area_unit_center[0] - point_reading_area_center[0]      # x轴正负， 大于0， 单位在读数区域右边，反之
    line_puc_prauc = point_unit_center[1] - point_read_area_unit_center[1]              # y轴正负，大于0，读数区域在单位区域上方，反之

    angle_h = math.atan(line_puc_prauc/line_prauc_prac)                # angle_h 斜率


    if line_puc_prauc > 0 and line_prauc_prac > 0:              # 第四象限
        return angle_h * 180 / math.pi
    elif line_puc_prauc < 0 and line_prauc_prac > 0:           # 第一象限
        return angle_h * 180 / math.pi
    elif line_puc_prauc > 0 and line_prauc_prac < 0:             # 第三象限
        return 180 + angle_h * 180 / math.pi
    elif line_puc_prauc < 0 and line_prauc_prac < 0:            # 第二象限
        return 180 + angle_h * 180 / math.pi

    elif line_puc_prauc == 0 and line_prauc_prac > 0:
        return 0
    elif line_prauc_prac == 0 and line_puc_prauc > 0:
        return 90
    elif line_puc_prauc == 0 and line_prauc_prac < 0:
        return 180
    elif line_prauc_prac == 0 and line_puc_prauc < 0:
        return 270
    else:
        pass


def get_angle_waterdial_readarea(point_water_dial_center, point_reading_area_center, point_waterdial_readarea_center):

    """                       (line_prac_pwrc)
                (pwrc )* -----------------------------*(prac)
                       |                             |
       (line_pwrc_pwdc)|                             |
                       |                             |
                       |                             |
                       |                             |
                (pwdc) * ----------------------------|
    Args:
        point_water_dial_center (pwdc): 水表盘中点 [x, y]
        point_reading_area_center (prac): 读数区中点 [x, y]
        point_waterdial_readarea_center (pwrc): 与读数区水平，与水表中心垂直 [x, y]

    Returns: 矫正角度

    """

    line_prac_pwrc = point_reading_area_center[0] - point_water_dial_center[0]             # x 轴方向，
    line_pwrc_pwdc = point_water_dial_center[1] - point_waterdial_readarea_center[1]       # y 轴方向。

    angle_h = math.atan(line_pwrc_pwdc/line_prac_pwrc)

    if line_prac_pwrc > 0 and line_pwrc_pwdc >0:    # 一象限           正确
        return 90 - angle_h * 180 / math.pi
    elif line_prac_pwrc < 0 and  line_pwrc_pwdc > 0:   # 二象限        正确
        return abs(angle_h * 180 / math.pi) + 270
    elif line_prac_pwrc > 0 and line_pwrc_pwdc < 0:   # 四象限         正确
        return abs(angle_h * 180 / math.pi) + 90
    elif line_prac_pwrc < 0 and line_pwrc_pwdc < 0:   # 三象限         正确
        return 180 + 90 - angle_h * 180 / math.pi

    elif line_prac_pwrc == 0 and line_pwrc_pwdc > 0:
        return 0
    elif line_pwrc_pwdc == 0 and line_prac_pwrc > 0:
        return 90
    elif line_prac_pwrc == 0 and line_pwrc_pwdc < 0:
        return 180
    elif line_pwrc_pwdc == 0 and line_prac_pwrc < 0:
        return 270
    else:
        pass

if __name__ == '__main__':
    point_a = [46.93188214302063, 169.6258773803711]
    point_b = [105.89911270141602, 212.37411499023438]
    point_c = [105.89911270141602, 169.6258773803711]


    jiaodu = get_angle(point_a, point_b, point_c)
    print("旋转角度为：", jiaodu)


    import cv2
    import numpy as np

    src = cv2.imread("ces/3910441621570393052.jpg")
    cv2.imshow('s', src)
    rows, cols = src.shape[: 2]

    # 仿射变换
    M = cv2.getRotationMatrix2D(center=point_a, angle=jiaodu, scale=1)
    # M = cv2.getAffineTransform(post1, post2)

    res = cv2.warpAffine(src, M, (cols, rows))
    cv2.imshow('res', res)
    cv2.imwrite('res', res)
    cv2.waitKey(0)


    # 透视变换
    # resse = cv2.imread()
    # srce = np.float32([[0, 50], [200, 50], [0, 378], [200, 378]])
    # dst = np.float32([0, 0], [378, 0], [0, 378], [378, 378])
    # M = cv2.getPerspectiveTransform(srce, dst=dst)
    # new = cv2.warpPerspective(res, M, (378, 378))
    # cv2.imshow('or', new)
    # cv2.waitKey(0)

# -*- coding: utf-8 -*-
__author__ = 'joy'

import  os
import numpy as np
import datetime
import csv

POI_CLASS_NUM = 15
# 15类poi_type_name顺序
ptn = ['购物服务区','行政区','交通物流区','金融保险区','商务住宅区','工业区','科教区','公共服务区','风景名胜区','医疗保健区','生活服务区','住宿服务区','车辆服务区','餐饮服务区','休闲娱乐区']

def get_region_ids(path):
    fr = open(path, 'r')
    alllines = fr.readlines()

    data = []
    for eachl in alllines:
        eachl = eachl.split("\n")[0]
        d = []
        b_arr = eachl.split(",")
        for b in b_arr:
            b = (int)(b)
            d.append(b)
        data.append(d)
    return data


def read_region_type(region_ids):
    type_dic = {}
    with open("RealFunction_1007_4.csv", "r") as f:
        fr = csv.reader(f)
        for row in fr:
            id = (row[0].split("\t")[0])
            if id == "Area_ID":
                continue
            row_str = ''.join(row)
            type_str = (row_str.split("\t")[1])
            type_dic[id] = type_str
    return type_dic


def get_poi_vec(path):
    fr = open(path, 'r')
    allines = fr.readlines()
    sum = 0
    arr = np.zeros(POI_CLASS_NUM)
    for eachl in allines:
        a = eachl.split("\n")[0].split(",")
        type = (int)(a[0]) - 1
        sum += (int)(a[1])
        arr[type] = (int)(a[1])
    if sum > 0:
        vec = np.divide(arr, sum)
    else:
        vec = arr
    return sum, vec


def is_same_type(id, type, type_dic):
    acc_type_str = type_dic[str(id)]
    if acc_type_str.find(type) != -1:
        return True
    else:
        return False


def get_sciwk_res(src_path, regions, type_dic):
    region_num = len(regions)
    famouspoi_thres = 1
    fam_weight = 0.6
    sum_acc = 0
    sum_num = 0
    for r_inx in range(region_num):
        region_ids = regions[r_inx]
        poi_vec = np.zeros(POI_CLASS_NUM)
        acc_count = 0
        for inx in region_ids:
            inx = str(inx).strip("\n")
            d_file = os.path.join(src_path, "splitRoad",  str(inx) + ".dbf")
            f_name = os.path.basename(d_file).split(".")[0].split("\n")[0]

            fpoi_path = os.path.join(src_path, "POI_CLASS_In_Area", "FamousPOI_Class_In_Area", f_name + ".txt")
            fpoi_sum, fpoi = get_poi_vec(fpoi_path)

            chkpoi_path = os.path.join(src_path, "POI_CLASS_In_Area", "CheckPOI_Class_In_Area", f_name + ".txt")
            _, chkpoi = get_poi_vec(chkpoi_path)

            if fpoi_sum >= famouspoi_thres: #famous poi数量阈值 可调
                score_vec =  fam_weight * fpoi * 100  + (1 - fam_weight) * chkpoi * 100
            else:
                score_vec = chkpoi * 100

            type_inx = np.argmax(score_vec)
            poi_vec[type_inx] += 1

            if is_same_type((int)(inx), ptn[type_inx], type_dic):
                acc_count += 1

        acc = round(acc_count * 1.0 / len(region_ids), 2)
        sum_acc += acc
        poi_type_inx = np.argmax(poi_vec)
        print("第{}号类簇".format(str(r_inx)), "类别是:", ptn[poi_type_inx], "准确率为：", acc)

    avg_acc = round(sum_acc / 9, 2)
    print("200个区域标注准确率为：", avg_acc)

if __name__ == "__main__":
    src_path = "SICWK"
    res_path = os.path.join(src_path, "200_FunctionArea.txt")
    # 获取每一类簇的区域ID
    region_ids = get_region_ids(res_path)
    # 获取每一区域对应的人工标注
    type_dic = read_region_type(region_ids)
    # 计算每一类簇类别
    get_sciwk_res(src_path, region_ids, type_dic)


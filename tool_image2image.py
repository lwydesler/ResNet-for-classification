from osgeo import gdal
import numpy as np
from PIL import Image
from PIL import ImageEnhance
import os
import codecs

#裁剪后栅格地质图重新赋坐标，alt_adress:裁剪后栅格地质图地址， result_j:对应区域遥感影像地址
def Image_transform(alt_adress, result_j):
    alt = gdal.Open(alt_adress)
    alt1 = alt.GetRasterBand(1)
    im_width = alt.RasterXSize  # 栅格矩阵的列数
    im_height = alt.RasterYSize  # 栅格矩阵的行数
    im_geotrans = alt.GetGeoTransform()  # 仿射矩阵

    blt = gdal.Open(result_j)
    blt1 = blt.GetRasterBand(1)
    data = blt1.ReadAsArray(0, 0, im_width, im_height)

    im_proj = alt.GetProjection()  # 地图投影信息
    im_data = alt1.ReadAsArray(0, 0, im_width, im_height)
    im_bands, (im_height, im_width) = 1, im_data.shape

    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create('E:/geoclass/transform/clip_15_class.tiff', im_width, im_height, im_bands, gdal.GDT_Float64)
    dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    dataset.SetProjection(im_proj)  # 写入投影
    dataset.GetRasterBand(1).WriteArray(data)
    del dataset

#统计类别及其数量， class_adress:转换完毕后栅格地质图地址
def text_class_number(class_adress):

    dic = {}
    x = 0
    y = 0
    class_num = 0
    data_Statistics = []

    obj = gdal.Open(class_adress)
    obj1 = obj.GetRasterBand(1)
    im_width = obj.RasterXSize
    im_height = obj.RasterYSize
    data = obj1.ReadAsArray(0, 0, im_width, im_height)
    print(im_width, im_height)

    while y < im_height:
        while x < im_width:
            value = data[y, x]
            if value not in dic:
                dic[value] = class_num
                data_Statistics.append(0)
                class_num = class_num + 1
            data_Statistics[dic[value]] = data_Statistics[dic[value]] + 1

            x = x + 1
        y = y + 1
        x = 0
    print(dic)
    print(data_Statistics)

    return data, dic



def cut_label_and_data(hight, weight, strip_y, strip_x, class_adress):

    data, _ = text_class_number(class_adress)
    dic = {8: 0, 2: 1, 5: 2, 17: 3, 19: 4, 4: 5, 16: 3, 10: 6, 20: 7, 15: 8, 7: 9, 6: 255, 18: 3, 21: 3}
    data_input_name = "E:/geoclass/pre_process/resize_pca123_PART2_NOTIFF_FULL.png"
    data_output_name = "E:/geoclass/class/"

    data_img_open =Image.open(data_input_name)

    real_value = []
    n = 16151#16150
    #左上角切割
    x1 = 0
    y1 = 0
    x2 = weight
    y2 = hight

    #纵向
    while x2 <= data_img_open.size[1]:
        #横向切
        while y2 <= data_img_open.size[0]:
            dataset_name = str(n) + ".png"
            dataset = data_img_open.crop((y1, x1, y2, x2))
            value = data[(x2-25), (y2-25)]
            real_value.append(int(dic[value]))
            dataset.save(data_output_name + str(dic[value]) + '/' + dataset_name)

            y1 = y1 + strip_y
            y2 = y1 + weight
            n = n + 1

        x1 = x1 + strip_x
        x2 = x1 + hight
        y1 = 0
        y2 = weight

    with open('E:/geoclass/real_value_part2.txt', 'w') as file:
        for i in real_value:
            file.write(str(i) + '\n')
        file.close()

    print("图片切割成功，切割得到的子图片数为"+str(n-1))
    return n-1


#得到测试集所需图片及真值
def test_label_and_data(hight, weight, strip_y, strip_x, class_adress):

    data, dic1 = text_class_number(class_adress)
    dic = {8: 0, 2: 1, 5: 2, 17: 3, 19: 4, 4: 5, 16: 3, 10: 6, 20: 7, 15: 8, 7: 9, 6: 255, 18: 3, 21: 3}
    data_input_name = "E:/geoclass/pre_process/resize_pca123_PART2_NOTIFF_FULL.png"
    data_output_name = "E:/geoclass/test/test1/"

    data_img_open =Image.open(data_input_name)

    n = 1
    real_value = []
    #左上角切割
    x1 = 0
    y1 = 0
    x2 = weight
    y2 = hight

    #纵向
    while x2 <= data_img_open.size[1]:
        #横向切
        while y2 <= data_img_open.size[0]:
            dataset_name = str(n) + ".png"
            dataset = data_img_open.crop((y1, x1, y2, x2))
            value = data[(x2-25), (y2-25)]
            value1 = dic[value]
            print(value1)
            real_value.append(value1)
            dataset.save(data_output_name + dataset_name)

            y1 = y1 + strip_y
            y2 = y1 + weight
            n = n + 1

        x1 = x1 + strip_x
        x2 = x1 + hight
        y1 = 0
        y2 = weight

        with open('E:/geoclass/real_value_part2.txt', 'w') as file:
            for i in real_value:
                file.write(str(i)+'\n')
            file.close()

    print("图片切割成功，切割得到的子图片数为" + str(n-1))
    return n-1


#真值转换为tiff图像，a1, a2,真值所代表像元栅格分辨率，txt真值记录文件，im_width, im_height真值图的列行数
#tif_adress转换后geo_class图像
def read_text_to_array(text_adress, tif_adress):
    class_list = []
    with open(text_adress, 'r') as file:
        for line in file:
            for class_num in line.split(' '):
                print(class_num)
                class_list.append(class_num)
            print(class_list[-1])
    file.close()
    class_list1 = np.array(class_list)
    class_list2 = class_list1.reshape(112, 254)

    alt = gdal.Open(tif_adress)
    alt1 = alt.GetRasterBand(1)
    #im_width = alt.RasterXSize  # 栅格矩阵的列数
    #im_height = alt.RasterYSize  # 栅格矩阵的行数
    im_width, im_height = 254, 112
    a, a1, b, c, d, a2 = alt.GetGeoTransform()  # 仿射矩阵
    print(a, a1, b, c, d, a2)
    im_geotrans = (a, 50.0, b, c, d, -50.0)
    im_proj = alt.GetProjection()  # 地图投影信息
    im_data = alt1.ReadAsArray(0, 0, im_width, im_height)
    im_bands, (im_height, im_width) = 1, im_data.shape

    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create('E:/geoclass/test_result/test_reall.tiff', im_width, im_height, im_bands, gdal.GDT_Float64)
    dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    dataset.SetProjection(im_proj)  # 写入投影
    dataset.GetRasterBand(1).WriteArray(class_list2)
    del dataset

def text_predict(text_predict_adress, test_reall_tiff_adress):
    dic = {8.0: 0, 4.0: 1, 17.0: 2, 5.0: 3, 19.0: 4, 7.0: 5, 16.0: 6, 2.0: 7, 25.0: 8, 10.0: 9, 20.0: 10, 15.0: 11}
    dic_re = {8: 0, 2: 1, 5: 2, 17: 3, 19: 4, 4: 5, 16: 3, 10: 6, 20: 7, 15: 8, 7: 9, 6: 255, 18: 3, 21: 3}
    class_list = []
    with open(text_predict_adress, 'r') as file:
        for line in file:
            print(line)
            class_list.append(int(line))
        print(class_list[-1])
    file.close()

    alt = gdal.Open(test_reall_tiff_adress)
    alt1 = alt.GetRasterBand(1)
    im_width = alt.RasterXSize  # 栅格矩阵的列数
    im_height = alt.RasterYSize  # 栅格矩阵的行数

    class_list1 = np.array(class_list)
    class_list2 = class_list1.reshape(70, 129)#70，129 / 95, 170

    a, a1, b, c, d, a2 = alt.GetGeoTransform()
    im_geotrans = (a, 375.0, b, c, d, -375.0)  # 仿射矩阵
    im_proj = alt.GetProjection()  # 地图投影信息
    im_data = alt1.ReadAsArray(0, 0, im_width, im_height)
    im_bands, (im_height, im_width) = 1, im_data.shape

    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create('E:/geoclass/test_result/real_part2.tiff', 129, 70, im_bands, gdal.GDT_Float64)
    dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    dataset.SetProjection(im_proj)  # 写入投影
    dataset.GetRasterBand(1).WriteArray(class_list2)
    del dataset

def predict_txt(predict_txt_adress, test_reall_tiff_adress):
    class_list = []
    with open(predict_txt_adress, 'r') as file:
        for line in file:
            print(line)
            class_list.append(line)
        print(class_list[-1])
    file.close()
    class_list1 = np.array(class_list)
    class_list2 = class_list1.reshape(159, 519)

    alt = gdal.Open(test_reall_tiff_adress)
    alt1 = alt.GetRasterBand(1)
    #im_width = alt.RasterXSize  # 栅格矩阵的列数
    #im_height = alt.RasterYSize  # 栅格矩阵的行数

    im_width, im_height = 519, 159
    a, a1, b, c, d, a2 = alt.GetGeoTransform()  # 仿射矩阵
    print(a, a1, b, c, d, a2)
    im_geotrans = (a, 375.0, b, c, d, -375.0)  # 仿射矩阵

    im_proj = alt.GetProjection()  # 地图投影信息
    im_data = alt1.ReadAsArray(0, 0, im_width, im_height)
    im_bands, (im_height, im_width) = 1, im_data.shape

    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create('E:/geoclass/test_result/predict_25.tiff', im_width, im_height, im_bands, gdal.GDT_Float64)
    dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    dataset.SetProjection(im_proj)  # 写入投影
    dataset.GetRasterBand(1).WriteArray(class_list2)
    del dataset

def cut_data(hight, weight, strip_y, strip_x):

    #data, dic = text_class_number(class_adress)
    data_input_name = "E:/geoclass/pre_process/resize_pca123_PART2_NOTIFF_FULL.png"
    data_output_name = "E:/geoclass/test_10/"

    data_img_open =Image.open(data_input_name)

    n = 1
    real_value = []
    #左上角切割
    x1 = 0
    y1 = 0
    x2 = weight
    y2 = hight

    #纵向
    while x2 <= data_img_open.size[1]:
        #横向切
        while y2 <= data_img_open.size[0]:
            dataset_name = str(n) + ".png"
            dataset = data_img_open.crop((y1, x1, y2, x2))
            dataset.save(data_output_name + dataset_name)

            y1 = y1 + strip_y
            y2 = y1 + weight
            n = n + 1

        x1 = x1 + strip_x
        x2 = x1 + hight
        y1 = 0
        y2 = weight

    print("图片切割成功，切割得到的子图片数为")
    return n-1

#删除目录下文件，删除total_num * 1/2张
def random_remove(class_adress):

    list_name = []
    n = 1
    for file in os.listdir(class_adress):
        file_name = class_adress + '/' + file
        if n%3 == 0:
            print(n)
            os.remove(file_name)
        else:
            list_name.append(file_name)
            list_name.sort()
        n = n + 1
    print(list_name[0:20])

#随机增类别图片
def enhance_class_png(class_adress):
    k = 1
    n = 221000
    for file in os.listdir(class_adress):
        file_name = class_adress + '/' + file
        image = Image.open(file_name)
        # 色度增强
        if k % 2 == 0:
            enh_col = ImageEnhance.Color(image)
            color = 1.5
            image_colored = enh_col.enhance(color)
            image_colored.save(class_adress + '/' + str(n) +'.png')
            n += 1
            k += 1

        # 锐度增强
        else:
            enh_sha = ImageEnhance.Sharpness(image)
            sharpness = 3.0
            image_sharped = enh_sha.enhance(sharpness)
            image_sharped.save(class_adress + '/' + str(n) +'.png')
            n += 1
            k += 1



if __name__ == "__main__":
    alt_adress = "E:/geoclass/pca/geo_pca123_clip.tif"
    result_j = 'E:/geoclass/geo_raster/geo_class_15_clip.tif'
    class_adress = "E:/geoclass/transform/clip_15_class.tiff"
    text_adress = 'E:/geoclass/real_value.txt'
    predict_reall_tif = 'E:/geoclass/pre_process/GEO_CLASS_TIFF_ALL.tif'
    predict_txt_adress = 'E:/geoclass/predict_value_25.txt'
    #text_class_number(class_adress)
    #num = cut_label_and_data(50, 50, 25, 25, class_adress)
    #num = test_label_and_data(50, 50, 25, 25, class_adress)
    #read_text_to_array(text_adress, class_adress)
    #print(num)
    cut_data(50, 50, 10, 10)

    text_predict_adress = 'E:/geoclass/test_value_part1.txt'
    test_reall_tiff_adress = 'E:/geoclass/pre_process/geo_class_part1.tif'
    #text_predict(text_predict_adress, test_reall_tiff_adress)

    part1_adress = "E:/geoclass/pre_process/geo_class_part1.tif"
    part2_adress = "E:/geoclass/pre_process/geo_class_resizepart2_full.tif"

    text_real_adress = 'E:/geoclass/real_value_part2.txt'
    test_real_tiff_adress = 'E:/geoclass/pre_process/geo_class_part2.tif'

    #cut_label_and_data(50, 50, 25, 25, part2_adress)
    #random_remove("E:/geoclass/class/4")
    #enhance_class_png("E:/geoclass/class/9")

    #test_label_and_data(50, 50, 25, 25, part2_adress)
    #text_predict(text_predict_adress, test_reall_tiff_adress)
    #text_predict(text_real_adress, test_real_tiff_adress)
    #predict_txt(predict_txt_adress, test_reall_tiff_adress)


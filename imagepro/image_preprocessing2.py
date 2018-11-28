# coding=utf-8
from PIL import Image, ImageEnhance, ImageFilter
import os
import shutil


CONTRAST_FLAGE = True                     # 对比度标志位
LIGHT_FLAGE = True                        # 亮度增强标志位
COLOR_FLAGE = True                        # 色彩增强标志位
MIRROR_FLAGE = False                       # 镜像标志位
TRANSPOSE_FLAGE = True                    # 旋转图像标志位
CUT_FLAGE = False                          # 裁剪标志位
GAUSSIAN_BLUR = True                      # 高斯模糊标志位


'''高斯模糊'''
class MyGaussianBlur(ImageFilter.Filter):

    name = "GaussianBlur"

    def __init__(self, radius=2, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)


work_path = 'test'                # 存放要工作文件夹名
work_dir_name = '1'                       # 存放图像文件的文件夹名
convert_image_dir_name = 'test_convert_image'  # 存放转化后图片的文件夹名


cwd = os.getcwd()                                                                    # 取得当前路径
work_dir_path1 = cwd + os.sep + work_path + os.sep                    # 得到存放图像文件的文件夹路径
convert_image_dir_path1 = cwd + os.sep + convert_image_dir_name + os.sep # 得到存放转化后的图像文件的文件夹路径
# print work_dir_path

for j in range(1,2):
    work_dir_path= work_dir_path1
    convert_image_dir_path=convert_image_dir_path1
    print j

    image_number = 0                                      # 转换图片的数量
    if os.path.exists(convert_image_dir_path):            # 如果convert_image文件夹存在，则递归地删除convert_image文件夹
        shutil.rmtree(convert_image_dir_path)
    os.mkdir(convert_image_dir_path)                      # 生成convert_image文件夹
    image_file_name = os.listdir(work_dir_path)           # 得到每个图像文件的文件名list
    image_file_name_path = []                             # 保存要转化图像的绝对路径的list

    for x in image_file_name:
        image_file_name_path.append(work_dir_path + os.sep + x)                        # 得到每个图像文件的绝对路径list

    for x in image_file_name_path:
        ii = 1
        img = Image.open(x)                                                            # 打开图像，得到Image对象

        if MIRROR_FLAGE:                                                               # 对图像进行镜像
            img_mirror = img.transpose(Image.FLIP_LEFT_RIGHT)
            img_mirror.save(convert_image_dir_path + os.sep + 'mirror_' + os.path.basename(x))
        if CUT_FLAGE:                                                                  # 对图像进行裁剪
            img_cut = img.crop((100, 100, 350, 350))
            img_cut.save(convert_image_dir_path + os.sep + 'cut_' + os.path.basename(x))
        if GAUSSIAN_BLUR:
            img_gb = img.filter(MyGaussianBlur(radius=4))                              # 对图像进行高斯模糊，模糊半径为4
            img_gb.save(convert_image_dir_path + os.sep + os.path.basename(x).split('.')[0] + str(ii) + '.jpg')


        if TRANSPOSE_FLAGE:                                                            # 对图像进行旋转
            img_rotate_20 = img.rotate(20)                                             # 对图像旋转30度
            img_rotate_negative_30 = img.rotate(-20)                                   # 对图像旋转负30度
            img_rotate_20.save(convert_image_dir_path + os.sep + os.path.basename(x).split('.')[0] + str(ii+1) + '.jpg')
            img_rotate_negative_30.save(convert_image_dir_path + os.sep + os.path.basename(x).split('.')[0] + str(ii+2) + '.jpg')

            if j == 6:
                img_rotate_30 = img.rotate(30)
                img_rotate_30.save(convert_image_dir_path + os.sep + 'rotate_30_' + os.path.basename(x))

        if LIGHT_FLAGE:                                                                # 对图像进行亮度增强
            enh_bri = ImageEnhance.Brightness(img)
            brightness = 1
            img_brightened = enh_bri.enhance(brightness)
            img_brightened.save(convert_image_dir_path + os.sep + os.path.basename(x).split('.')[0] + str(ii+3) + '.jpg')

        if COLOR_FLAGE:                                                                # 对图像进行色彩增强
            enh_col = ImageEnhance.Color(img)
            color = 2
            img_colored = enh_col.enhance(color)
            img_colored.save(convert_image_dir_path + os.sep + os.path.basename(x).split('.')[0] + str(ii+4) + '.jpg')

        if CONTRAST_FLAGE:                                                             # 对图像进行对比度增强
            enh_con = ImageEnhance.Contrast(img)
            contrast = 2
            img_contrast = enh_con.enhance(contrast)
            img_contrast.save(convert_image_dir_path + os.sep + os.path.basename(x).split('.')[0] + str(ii+5) + '.jpg')

        image_number += 1
        # 显示处理到第几张,尺寸，图像模式
        # print j
        # print("convert pictur" "es :%s size:%s mode:%s" % (image_number, img.size, img.mode))

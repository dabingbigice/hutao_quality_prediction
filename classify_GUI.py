import math
import os
import time
import tkinter.ttk
from datetime import datetime
from tkinter import *
from tkinter import filedialog

import joblib

from ellipse_fitting_img import single_file_predict as ellipse_fitting
from ellipse_fitting_img import single_file_predict_online as ellipse_fitting_online
import cv2
import torch
from torchvision import transforms
import pandas as pd
from torchvision import models
import numpy as np
from PIL import Image, ImageTk
import tkinter.font as tkFont
from tkinter import ttk
import json
from PIL import Image, ImageTk
from model.deeplab import DeeplabV3
from perimeter import hutao_perimeter
import pickle

# 全局背景图片引用
bg_image = None
bg_photo = None
my_class = ["background", "hutao_all", "walnut_half"]

deeplab = DeeplabV3()


# 加载保存的SVR模型
# 加载模型和标准化器（假设您保存了scaler）
def load_artifacts(model_path, scaler_path):
    with open(model_path, 'rb') as f_model, open(scaler_path, 'rb') as f_scaler:
        model = joblib.load(f_model)
        scaler = joblib.load(f_scaler)
    return model, scaler


model, scaler = load_artifacts(f"./svr_model_pkl/model_svr_seed3.pkl", f"./svr_model_pkl/scaler_svr_seed3.pkl")  # 修改路径


def calculate_aspect_ratio(perimeter, area):
    # 计算长轴 L 和短轴 W
    discriminant = perimeter ** 2 - 16 * area
    if discriminant < 0:
        return None  # 无实数解（可能不是矩形）

    sqrt_discriminant = math.sqrt(discriminant)
    L1 = (perimeter + sqrt_discriminant) / 4
    L2 = (perimeter - sqrt_discriminant) / 4

    # 取较大的值作为 L（长轴）
    L = max(L1, L2)
    print(f'长:{L}')
    W = area / L  # 因为 A = L × W
    print(f'宽:{W}')

    aspect_ratio = L / W
    return L, W, aspect_ratio


import numpy as np
from scipy.optimize import fsolve
from scipy.special import ellipe


def ellipse_perimeter(a, b):
    if a < b:
        a, b = b, a  # 确保 a >= b
    e_sq = 1 - (b ** 2 / a ** 2)
    return 4 * a * ellipe(e_sq)


def ellipse_axes_from_area_perimeter(A, P):
    # 检查最小周长
    min_P = 2 * np.pi * np.sqrt(A / np.pi)
    if P < min_P:
        raise ValueError(f"周长 P = {P} 太小，至少需要 {min_P:.2f}")

    def equations(vars):
        a, b = vars
        eq1 = np.pi * a * b - A
        eq2 = ellipse_perimeter(a, b) - P
        return [eq1, eq2]

    # 初始猜测：假设 a >= b
    initial_a = np.sqrt(A)
    initial_b = A / (np.pi * initial_a)
    a, b = fsolve(equations, (initial_a, initial_b))

    return a, b, a / b


def set_background(root, image_path):
    global bg_image, bg_photo
    try:
        # 加载并调整背景图片
        bg_image = Image.open(image_path)
        bg_image = bg_image.resize((root.winfo_screenwidth(), root.winfo_screenheight()), Image.LANCZOS)
        bg_photo = ImageTk.PhotoImage(bg_image)

        # 创建背景画布
        bg_canvas = Canvas(root, highlightthickness=0)
        bg_canvas.create_image(0, 0, image=bg_photo, anchor="nw")
        bg_canvas.place(x=0, y=0, relwidth=1, relheight=1)

        # 窗口缩放事件绑定
        root.bind("<Configure>", lambda e: resize_background(root))
        return bg_canvas
    except Exception as e:
        print(f"背景加载失败: {str(e)}")
        return None


# 在全局变量区域新增
CAPTURE_DIR = os.path.join(os.path.dirname(__file__), "captured_images")  # 存储目录
os.makedirs(CAPTURE_DIR, exist_ok=True)  # 自动创建目录

is_cap = False

from excel_data import append_to_excel

save_dir_captured_orign = "captured_orign_photos"


# 新增拍照功能
def capture_image():
    global current_frame
    is_camera_running = True
    if is_camera_running:

        # 设置亮度,保存破损度分割数据照片
        if cap.get(cv2.CAP_PROP_BRIGHTNESS) != -1:
            # 获取当前亮度值（通常范围0-1或0-100，具体取决于驱动）
            current_brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
            # 尝试设置新亮度（示例值，需根据实际范围调整）
            target_brightness = 41
            cap.set(cv2.CAP_PROP_BRIGHTNESS, target_brightness)

        ret, frame = cap.read()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
        filename = f"capture_{timestamp}.jpg"
        save_path = os.path.join(save_dir_captured_orign, filename)
        # 保存图像（质量参数100%）
        save_success = cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])

        if save_success:
            print(f"图片保存成功：{os.path.abspath(save_path)}")
        else:
            print(f"错误：保存失败！检查路径权限：{save_path}")

        # 修改回分割模型需要的亮度
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 64)
        ret, frame = cap.read()
        fps = 0.0

        if ret:
            frame = cv2.resize(frame, (320, 320))
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            t1 = time.time()
            # 进行检测
            img, text, ratio, class_flag, area_num = deeplab.detect_image(frame, count=True, name_classes=my_class)

            # print(f'检测结果,text={text}\n,ratio={ratio},\n class_flag={class_flag}\n')

            # class_flag：0是背景，1是all,2是half,3是other
            t2 = time.time()
            # TODO 检测完成之后开启另外一个线程去显示画面。
            delta_ms = (t2 - t1) * 1000
            # print(f"deeplab.detect_image检测速度: {delta_ms:.3f} 毫秒")
            frame = np.array(img)

            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            fps = (fps + (1. / (t2 - t1))) / 2
            fps = (fps + (1. / (time.time() - t1))) / 2
            # print("fps= %.2f" % (fps))
            # frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            var.set(f'{my_class[class_flag]}:{ratio:.2f}%')
            var_area.set(f'{area_num:.2f}px')

            # 动态适配画布尺寸（显示用）
            img = Image.fromarray(frame)
            img = img.resize((320, 320), Image.LANCZOS)

            # 转换为OpenCV格式（保存用）
            save_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            save_img = np.ascontiguousarray(save_img, dtype=np.uint8)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            save_path = os.path.join(CAPTURE_DIR, filename)
            # 保存图像
            cv2.imwrite(save_path, save_img)

            # 更新画布
            photo = ImageTk.PhotoImage(image=img)
            show_img_1.delete("all")
            show_img_1.create_image(0, 0, anchor="nw", image=photo)
            show_img_1.image = photo  # 保持引用

            # 周长,保存标注后的周长掩码
            # perimeter = hutao_perimeter(save_path)
            # 计算椭圆a,b
            _, a, b, perimeter, error = ellipse_fitting(save_path, area_num)

            # print(f'perimeter={perimeter}')
            var_perimeter.set(f'{perimeter:.2f}px')
            circularity = area_num / perimeter
            # print(f"面积/周长: {circularity:.2f}")

            var_circularity.set(f'{circularity:.2f}')
            var_aspect_ratio.set(f'A={a:.1f},B={b:.1f},/={a / b:.1f}')

            value = var_input.get()  # 直接获取输入值
            g = float(value) / 100  # 转换为浮点数

            # 核桃仁h
            h = 0.58

            # 椭圆e
            a, b = max(a, b), min(a, b)  # 自动交换确保a >= b
            ratio_squared = (b / a) ** 2
            e = math.sqrt(1 - ratio_squared)

            # 核桃仁标定面积
            hutao_area = 0.00068 * area_num

            # 核桃仁标定周长
            hutao_perimeter = perimeter * 0.02596

            # 核桃仁标定周长面积比
            hutao_area_div_hutao_perimeter = hutao_area / hutao_perimeter

            # 拟合椭圆长半轴标定长
            hutao_a = a * 0.02596

            # 拟合椭圆短半轴标定长
            hutao_b = b * 0.02596

            # 长短轴比
            hutao_a_div_b = hutao_a / hutao_b

            # abh算术平均值
            arithmetic_a_b_h_avg = (hutao_a + hutao_b + h) / 3

            # abh几何平均值
            geometry_a_b_h_avg = (hutao_a * hutao_b * h) ** (1 / 3)

            # 形状索引
            hutao_SI = 2 * hutao_a / (h + hutao_b)

            # 厚度方向的伸长
            hutao_ET = hutao_a / h

            # 垂直方向的伸长
            hutao_EV = hutao_b / h

            # 球形度
            fai = (geometry_a_b_h_avg / hutao_a) * 100

            # 焦距
            hutao_c = np.sqrt(hutao_a ** 2 - hutao_b ** 2)

            # ab算术均值
            arithmetic_a_b_avg = (hutao_a + hutao_b) / 2
            # ab几何均值
            geometry_a_b_avg = (hutao_a * hutao_b) ** (1 / 2)

            new_data = [
                [area_num, perimeter, a, b, a / b, area_num / perimeter, g, error,
                 e, hutao_area, hutao_perimeter, hutao_area_div_hutao_perimeter, hutao_a,
                 hutao_b, hutao_a_div_b, arithmetic_a_b_h_avg, geometry_a_b_h_avg, hutao_SI,
                 hutao_ET, hutao_EV, fai, filename, hutao_c, arithmetic_a_b_avg, geometry_a_b_avg
                 ]

            ]
            append_to_excel('核桃仁表型信息_重新标定.xlsx', new_data)
            # 重置
            var_input.set(str(''))
    if current_frame is not None:
        # 生成唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"
        save_path = os.path.join(CAPTURE_DIR, filename)

        # 转换颜色空间并保存
        img = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img)

        # 打印日志（可选）
        print(f"图像已保存至：{save_path}")
        var.set(f"已保存：{filename}")


def resize_background(root):
    global bg_image, bg_photo
    if bg_image:
        # 动态适配窗口尺寸
        new_size = (root.winfo_width(), root.winfo_height())
        resized_img = bg_image.resize(new_size, Image.LANCZOS)
        bg_photo = ImageTk.PhotoImage(resized_img)
        root.children["!canvas"].itemconfig(1, image=bg_photo)  # 更新画布图片


with torch.no_grad():
    net_list = {'deeplab': deeplab,
                # 'resnet50': models.resnet50(pretrained=True),

                }

    trans = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_file = None
    path = None

    root = Tk()
    root.title('flower_predict')
    root.geometry("800x800")
    # 设置半透明样式
    style = ttk.Style()
    style.theme_create("transparent", settings={
        "TCombobox": {
            "configure": {"fieldbackground": "#FFFFFF80", "background": "#FFFFFF"},
            "map": {"background": [("readonly", "#FFFFFF80")]}
        }
    })
    # 在全局变量区域新增
    cap = None  # 摄像头对象
    is_camera_running = False  # 摄像头状态标志
    current_frame = None  # 当前帧缓存


    def open_pred():
        global cap, is_camera_running, current_frame
        if not is_camera_running:
            # 初始化摄像头
            cap = cv2.VideoCapture(1)
            # 设置分辨率
            target_width, target_height = 2048, 1536
            success_width = cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
            success_height = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)

            # 检查分辨率是否设置成功
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if (actual_width, actual_height) == (target_width, target_height):
                print(f"分辨率已设置为 {actual_width}x{actual_height}")
            else:
                print(f"警告：摄像头不支持 {target_width}x{target_height}，实际分辨率 {actual_width}x{actual_height}")

            is_camera_running = True
            update_camera_frame()  # 开始更新画面
        else:
            # 关闭摄像头
            cap.release()
            is_camera_running = False
            show_img_1.delete("all")  # 清空画布


    def pred():
        if current_frame is not None:
            # 转换帧为模型输入格式
            img = Image.fromarray(current_frame)
            img_tensor = trans(img).unsqueeze(0)

            # 执行预测（保持原有逻辑）
            net = net_list[bbox.get()]
            prediction = torch.softmax(net(img_tensor), dim=1)
            label = np.argmax(prediction.detach().numpy())


    def update_camera_frame():
        global current_frame
        if is_camera_running:
            ret, frame = cap.read()
            # 检查是否设置成功
            # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            # print(f"设置后的分辨率: {int(width)}x{int(height)}")
            # print(frame.shape)
            fps = 0.0
            if ret:
                frame = cv2.resize(frame, (320, 320))
                # 格式转变，BGRtoRGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 转变成Image
                frame = Image.fromarray(np.uint8(frame))
                t1 = time.time()
                # 进行检测
                img, text, ratio, class_flag, area_num = deeplab.detect_image(frame, count=True, name_classes=my_class)

                # print(f'检测结果,text={text}\n,ratio={ratio},\n class_flag={class_flag}\n')

                # class_flag：0是背景，1是all,2是half,3是other
                t2 = time.time()
                # TODO 检测完成之后开启另外一个线程去显示画面。
                delta_ms = (t2 - t1) * 1000
                # print(f"deeplab.detect_image检测速度: {delta_ms:.3f} 毫秒")
                frame = np.array(img)

                # RGBtoBGR满足opencv显示格式
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                fps = (fps + (1. / (t2 - t1))) / 2
                fps = (fps + (1. / (time.time() - t1))) / 2
                # print("fps= %.2f" % (fps))
                # frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                var.set(f'{my_class[class_flag]}:{ratio:.2f}%')
                var_area.set(f'{area_num:.2f}px')

                # 动态适配画布尺寸（显示用）
                img = Image.fromarray(frame)
                img = img.resize((320, 320), Image.LANCZOS)

                # 转换为OpenCV格式（保存用）
                save_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                save_img = np.ascontiguousarray(save_img, dtype=np.uint8)
                filename = f"capture.jpg"
                save_path = os.path.join(CAPTURE_DIR, filename)
                # 保存图像
                cv2.imwrite(save_path, save_img)

                # 更新画布
                photo = ImageTk.PhotoImage(image=img)
                show_img_1.delete("all")
                show_img_1.create_image(0, 0, anchor="nw", image=photo)
                show_img_1.image = photo  # 保持引用

                # 周长,保存标注后的周长掩码
                # perimeter = hutao_perimeter(save_path)
                # 计算椭圆a,b
                result, a, b, perimeter, error, x, y = ellipse_fitting_online(save_path, area_num)
                if result:
                    # print(f'perimeter={perimeter}')
                    var_perimeter.set(f'{perimeter:.2f}px')
                    circularity = area_num / perimeter
                    # print(f"面积/周长: {circularity:.2f}")

                    var_circularity.set(f'{circularity:.2f}')
                    var_aspect_ratio.set(f'A={a:.1f},B={b:.1f},/={a / b:.1f}')

                    # 核桃仁h
                    h = 0.84

                    # 椭圆e
                    a, b = max(a, b), min(a, b)  # 自动交换确保a >= b
                    ratio_squared = (b / a) ** 2

                    # 核桃仁标定面积
                    hutao_area = 0.00068 * area_num

                    # 核桃仁标定周长
                    hutao_perimeter = perimeter * 0.02596

                    # 拟合椭圆长半轴标定长
                    hutao_a = a * 0.02596

                    # 拟合椭圆短半轴标定长
                    hutao_b = b * 0.02596

                    # abh算术平均值
                    arithmetic_a_b_h_avg = (hutao_a + hutao_b + h) / 3

                    # abh几何平均值
                    geometry_a_b_h_avg = (hutao_a * hutao_b * h) ** (1 / 3)

                    # 形状索引
                    hutao_SI = 2 * hutao_a / (h + hutao_b)

                    # 厚度方向的伸长
                    hutao_ET = hutao_a / h

                    # 垂直方向的伸长
                    hutao_EV = hutao_b / h

                    # 球形度
                    fai = (geometry_a_b_h_avg / hutao_a) * 100

                    # ab算术均值
                    arithmetic_a_b_avg = (hutao_a + hutao_b) / 2
                    # ab几何均值
                    geometry_a_b_avg = (hutao_a * hutao_b) ** (1 / 2)

                    FEATURES = [hutao_area, hutao_perimeter, hutao_area / hutao_perimeter, hutao_a, hutao_b,
                                arithmetic_a_b_h_avg, geometry_a_b_h_avg, hutao_SI, hutao_ET, hutao_EV, fai,
                                arithmetic_a_b_avg, geometry_a_b_avg]

                    # 2. 转换输入数据
                    input_data = np.array(FEATURES).reshape(1, -1)

                    # 3. 标准化处理（关键步骤！）
                    scaled_data = scaler.transform(input_data)

                    # 4. 预测
                    prediction = model.predict(scaled_data)

                    print("标准化后的输入数据:\n", scaled_data)
                    print("\n预测结果:", prediction[0])
                    var_input.set(str(prediction[0]))

                    # 以3g进行分类，但是以2.8克进行结算统计误差
                    if prediction[0] > 3 and class_flag == 1:
                        # 大于3g且为1/2仁
                        pass
                    elif prediction[0] < 3 and class_flag == 1:
                        # 小于3g且为1/2仁
                        pass
                    else:
                        # 1/4仁
                        pass

                # 重置

                else:
                    print(f'核桃仁未到达中心区域x={x},y={y},处理失败')
                    var_input.set(str(''))

            # 每10ms刷新一次（约100fps）
            show_img_1.after(10, update_camera_frame)


    style.theme_use("transparent")
    # 添加背景（替换为你的图片路径）
    bg_canvas = set_background(root, "bg.jpeg")
    var = StringVar()
    var_area = StringVar()
    var_perimeter = StringVar()
    var_circularity = StringVar()
    var_aspect_ratio = StringVar()

    open_Style = tkFont.Font(family="Lucida Grande", size=12)
    pre_Style = tkFont.Font(family="Bahnschrift SemiBold", size=16)
    # 停止启动
    open_img = Button(root, command=open_pred, font=open_Style)
    open_img.place(relx=0.85, rely=0.6, anchor='nw', relwidth=1 / 8, relheight=1 / 12)
    # 修改原"选择图片"按钮为摄像头开关
    open_img.config(text='启动/停止', command=open_pred)

    pred_button = Button(root, text='预测', command=pred, font=open_Style)
    pred_button.place(relx=0.75, rely=0.6, anchor='ne', relwidth=1 / 8, relheight=1 / 12)

    show_img_1 = Canvas(root, borderwidth=2, relief='sunken')
    show_img_1.place(relx=0.06, rely=1 / 15, anchor='nw', relwidth=0.4, relheight=0.4)

    show_img_2 = Canvas(root, borderwidth=2, relief='sunken')
    show_img_2.place(
        relx=0.06,  # 水平位置与show_img_1对齐 [3](@ref)
        rely=(1 / 15 + 0.433),  # 垂直位置计算：show_img_1起始位置 + 高度 + 间距 [3](@ref)
        anchor='nw',  # 保持左上角对齐 [3](@ref)
        relwidth=0.4,  # 宽度与show_img_1一致
        relheight=0.4  # 高度占窗口30% [4](@ref)
    )

    # 设置类别标签
    cls_label = Label(root, text='模型选择: ', font=pre_Style)
    cls_label.place(relx=0.53, rely=0.06, anchor='nw', relwidth=0.15, relheight=0.05)

    # 设置下拉框
    bbox = ttk.Combobox(root,
                        values=['deeplab', ], font=pre_Style)
    bbox.set('deeplab')
    bbox.place(relx=0.69, rely=0.06, anchor='nw', relwidth=0.3, relheight=0.05)

    pre_label = Label(root, text='预测结果: ', font=pre_Style)
    pre_label.place(relx=0.53, rely=0.13, anchor='nw', relwidth=0.15, relheight=0.05)
    pre_label_area = Label(root, text='面积: ', font=pre_Style)
    pre_label_area.place(relx=0.53, rely=0.2, anchor='nw', relwidth=0.15, relheight=0.05)

    pre_label_perimeter = Label(root, text='周长: ', font=pre_Style)
    pre_label_perimeter.place(relx=0.53, rely=0.27, anchor='nw', relwidth=0.15, relheight=0.05)

    pre_label_perimeter = Label(root, text='面积/周长: ', font=pre_Style)
    pre_label_perimeter.place(relx=0.53, rely=0.34, anchor='nw', relwidth=0.15, relheight=0.05)
    pre_label_perimeter = Label(root, text='长轴/短轴: ', font=pre_Style)
    pre_label_perimeter.place(relx=0.53, rely=0.41, anchor='nw', relwidth=0.15, relheight=0.05)

    # 显示预测概率
    predictive_probability = Label(root, textvariable=var, font=pre_Style, borderwidth=5, relief='groove')
    predictive_probability.place(relx=0.69, rely=0.13, anchor='nw', relwidth=0.3, relheight=0.05)

    predictive_area = Label(root, textvariable=var_area, font=pre_Style, borderwidth=5, relief='groove')
    predictive_area.place(relx=0.69, rely=0.2, anchor='nw', relwidth=0.3, relheight=0.05)

    predictive_perimeter = Label(root, textvariable=var_perimeter, font=pre_Style, borderwidth=5, relief='groove')
    predictive_perimeter.place(relx=0.69, rely=0.27, anchor='nw', relwidth=0.3, relheight=0.05)

    predictive_var_circularity = Label(root, textvariable=var_circularity, font=pre_Style, borderwidth=5,
                                       relief='groove')
    predictive_var_circularity.place(relx=0.69, rely=0.34, anchor='nw', relwidth=0.3, relheight=0.05)
    predictive_var_circularity = Label(root, textvariable=var_aspect_ratio, font=pre_Style, borderwidth=5,
                                       relief='groove')
    predictive_var_circularity.place(relx=0.69, rely=0.41, anchor='nw', relwidth=0.3, relheight=0.05)
    import tkinter as tk

    # 在现有变量定义后添加新的StringVar
    var_input = tk.StringVar()  # 新增变量用于绑定输入框


    # 输入验证函数
    def validate_float(input_str):
        """浮点数输入验证，最多允许一个小数点"""
        if input_str.count('.') <= 1 and (input_str.replace('.', '', 1).isdigit() or input_str == ""):
            return True
        return False


    input_label = tk.Label(root, text='滤波阈值: ', font=pre_Style)
    input_label.place(relx=0.53, rely=0.48, anchor='nw', relwidth=0.15, relheight=0.05)

    # 浮点数输入框
    vcmd = root.register(validate_float)  # 注册验证函数
    input_entry = tk.Entry(
        root,
        textvariable=var_input,
        validate="key",  # 实时验证输入
        validatecommand=(vcmd, '%P'),
        font=pre_Style,
        borderwidth=5,
        relief='groove'
    )
    input_entry.place(relx=0.69, rely=0.48, anchor='nw', relwidth=0.3, relheight=0.05)

    # 在按钮布局区域添加（放在预测按钮下方）
    capture_btn = Button(root, text='拍照保存', command=capture_image,
                         font=open_Style, bg="#4CAF50", fg="white")
    capture_btn.place(relx=0.75, rely=0.7, anchor='ne',
                      relwidth=1 / 8, relheight=1 / 12)
    root.bind('<Return>', lambda event: capture_image())
    root.mainloop()

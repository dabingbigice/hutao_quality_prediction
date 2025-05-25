import math
import os
import time
from datetime import datetime
from tkinter import *
from utils.gui_tools import set_background
from ellipse_fitting_img import single_file_predict as ellipse_fitting
from ellipse_fitting_img import single_file_predict_online as ellipse_fitting_online
import cv2
import torch
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tkinter.font as tkFont
from tkinter import ttk
from model.deeplab import DeeplabV3
from utils.util_tools import load_artifacts
from utils.cap_tools import set_cap_config
import threading
from excel_data import append_to_excel

thread_flag = True

# 开启多线程


# 调用示例

my_class = ["background", "hutao_all", "walnut_half"]

deeplab = DeeplabV3()


CAP_INDEX = f'E:\H 黄#雀 (2025)\\01.mp4'
CAP1_INDEX = f'E:\H 黄#雀 (2025)\\02.mp4'
# 加载保存的SVR模型
# 加载模型和标准化器（假设您保存了scaler）
model, scaler = load_artifacts(f"./svr_model_pkl/model_svr_seed3.pkl", f"./svr_model_pkl/scaler_svr_seed3.pkl")  # 修改路径
# 在全局变量区域新增
CAPTURE_DIR = os.path.join('F:', "captured_images")  # 存储目录
os.makedirs(CAPTURE_DIR, exist_ok=True)  # 自动创建目录
is_cap = False
# 在全局变量区域新增
cap = None  # 摄像头对象
is_camera_running = False  # 摄像头状态标志
is_camera_running1 = False  # 摄像头状态标志
current_frame = None  # 当前帧缓存
save_dir_captured_orign = "captured_orign_photos"
# 添加目录存在性检查（自动创建缺失目录）
os.makedirs(CAPTURE_DIR, exist_ok=True)


# 图像处理
def start_camera_thread(cap, canvas, is_running_flag, filename):
    """启动摄像头更新线程并返回线程ID"""
    # 创建线程对象
    thread = threading.Thread(
        target=update_camera_frame,
        args=(cap, canvas, is_running_flag, filename),
        name=str(filename),  # 可选：设置线程名称
        daemon=True
    )

    # 启动线程
    thread.start()

    # 返回线程ID（Python标识符和原生ID）
    return {
        "thread_object": thread,
        "ident": thread.ident,
        "native_id": thread.native_id
    }


# 新增拍照功能,用于提取表型信息
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


with torch.no_grad():
    net_list = {'GhostLab*': deeplab,
                # 'resnet50': models.resnet50(pretrained=True),
                }

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

    pred_button = Button(root, text='预测', command=None, font=open_Style)
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
    bbox.set('GhostLab*')
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


    input_label = tk.Label(root, text='预测重量(g): ', font=pre_Style)
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


    def img_process(frame, show_img, filename, current_cam_index):

        fps = 0.0
        t1 = time.time()
        frame = cv2.resize(frame, (320, 320))
        # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转变成Image
        frame = Image.fromarray(np.uint8(frame))
        # 进行检测
        img, text, ratio, class_flag, area_num = deeplab.detect_image(frame, count=True, name_classes=my_class)
        # print(f'检测结果,text={text}\n,ratio={ratio},\n class_flag={class_flag}\n')

        # class_flag：0是背景，1是all,2是half,3是other
        t2 = time.time()
        # print(f"deeplab.detect_image检测速度: {delta_ms:.3f} 毫秒")
        frame = np.array(img)

        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        fps = (fps + (1. / (t2 - t1))) / 2
        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %.2f" % (fps))
        # frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 动态适配画布尺寸（显示用）
        img = Image.fromarray(frame)
        img = img.resize((320, 320), Image.LANCZOS)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("arial.ttf", 20)
        draw.text((10, 10), f"FPS: {fps:.1f}", font=font, fill=(255, 0, 0))

        # 转换为OpenCV格式（保存用）
        save_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        save_img = np.ascontiguousarray(save_img, dtype=np.uint8)
        save_path = os.path.join(CAPTURE_DIR, filename)
        # 保存图像
        cv2.imwrite(save_path, save_img)
        # 更新画布
        photo = ImageTk.PhotoImage(image=img)
        show_img.delete("all")
        show_img.create_image(0, 0, anchor="nw", image=photo)
        show_img.image = photo  # 保持引用
        # 输入框显示
        if current_cam_index == 'cap.jpg':
            var.set(f'{my_class[class_flag]}:{ratio:.2f}%')
            var_area.set(f'{area_num:.2f}px')

        # 计算椭圆a, b, perimeter, error, x, y
        result, a, b, perimeter, error, x, y = ellipse_fitting_online(save_path, area_num)
        if result:
            # print(f'perimeter={perimeter}')

            circularity = area_num / perimeter
            # print(f"面积/周长: {circularity:.2f}")

            # 核桃仁h
            h = 0.84

            # 椭圆e
            a, b = max(a, b), min(a, b)  # 自动交换确保a >= b

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

            print("\n预测结果:", prediction[0])

            if current_cam_index == 'cap.jpg':
                var_perimeter.set(f'{perimeter:.2f}px')
                var_circularity.set(f'{circularity:.2f}')
                var_aspect_ratio.set(f'A={a:.1f},B={b:.1f},/={a / b:.1f}')
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
            print(f'cap={current_cam_index},核桃仁未到达中心区域x={x},y={y},处理失败')
            if current_cam_index == 'cap.jpg':
                var_input.set(str(''))


    def open_close_cap_pred():
        global cap, is_camera_running, current_frame, cap1, is_camera_running1, thread_flag
        if not is_camera_running:
            # 初始化摄像头
            cap = cv2.VideoCapture(CAP_INDEX)
            cap1 = cv2.VideoCapture(CAP1_INDEX)
            # 设置分辨率
            set_cap_config(cap)
            set_cap_config(cap1)

            is_camera_running = True
            is_camera_running1 = True
            # if thread_flag:
            #     thread_flag = False
            thread_object, ident, native_id = start_camera_thread(cap, show_img_1, is_camera_running, 'cap.jpg')
            thread_object, ident, native_id = start_camera_thread(cap1, show_img_2, is_camera_running1, 'cap_1.jpg')
            # update_camera_frame(cap, show_img_1, is_camera_running, 'cap.jpg')  # 开始更新画面
            # update_camera_frame(cap1, show_img_2, is_camera_running1, 'cap_1.jpg')  # 开始更新画面
        else:
            # 关闭摄像头
            cap.release()
            cap1.release()
            is_camera_running = False
            is_camera_running1 = False
            show_img_1.delete("all")  # 清空画布
            show_img_2.delete("all")  # 清空画布


    def update_camera_frame(cap_invoke, show_img, is_camera_running_invoke, filename):
        global current_frame
        while is_camera_running_invoke:
            ret1, frame1 = cap_invoke.read()
            if ret1:
                img_process(frame1, show_img, filename, str(filename))
            time.sleep(0.02)
        # 每10ms刷新一次（约100fps）
        # show_img.after(250, update_camera_frame, cap_invoke, show_img, is_camera_running_invoke, filename)
        # 停止启动
        cap_invoke.release()
        show_img.delete("all")  # 清空画布


    open_img = Button(root, command=open_close_cap_pred, font=open_Style)
    open_img.place(relx=0.85, rely=0.6, anchor='nw', relwidth=1 / 8, relheight=1 / 12)
    # 修改原"选择图片"按钮为摄像头开关
    open_img.config(text='启动/停止', command=open_close_cap_pred)
    root.mainloop()

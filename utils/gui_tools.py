from tkinter import *

from PIL import ImageTk
from PIL import Image

def resize_background(root):
    global bg_image, bg_photo
    if bg_image:
        # 动态适配窗口尺寸
        new_size = (root.winfo_width(), root.winfo_height())
        resized_img = bg_image.resize(new_size, Image.LANCZOS)
        bg_photo = ImageTk.PhotoImage(resized_img)
        root.children["!canvas"].itemconfig(1, image=bg_photo)  # 更新画布图片


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

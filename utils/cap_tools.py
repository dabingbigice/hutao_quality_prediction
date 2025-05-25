import cv2


def set_cap_config(cap):
    target_width, target_height = 2048, 1536
    # 设置高度和宽度
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
    # 检查分辨率是否设置成功
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"分辨率已设置为 {actual_width}x{actual_height}")
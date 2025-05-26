import serial
import binascii


class stm32Serial:
    def __init__(self, port, baudrate):
        self.ser = serial.Serial(port, baudrate, timeout=1)
        print(f'初始化stm32串口对象={self.ser}')

    def close_serial(self):
        self.ser.close()
        print(f'关闭串口结果={self.ser.is_open}')

    def send_to_stm32(self, message="0"):
        try:
            print(f"已连接 {self.ser.name}")
            # 核心修改：将 "0"/"1" 转换为 0x00/0x01
            # 0号控制位
            if message == "0":
                hex_data = b'\x00'  # 二进制0
            elif message == "1":
                hex_data = b'\x01'  # 二进制1
            #     1号控制位
            elif message == "2":
                hex_data = b'\x02'  # 二进制2
            elif message == "3":
                hex_data = b'\x03'  # 二进制3

            else:
                hex_data = message.encode('gbk')  # 其他内容仍用GBK编码

            self.ser.write(hex_data)
            print(f"发送HEX: {binascii.hexlify(hex_data).decode()}")

            # 接收响应（假设ESP32返回HEX）
            # time.sleep(0.5)
            # if ser.in_waiting > 0:
            #     response = ser.read_all()
            #     print(f"响应HEX: {binascii.hexlify(response).decode()}")

        except Exception as e:
            print(f"错误: {str(e)}")


def get_stm32_object(port):
    return stm32Serial(port=port, baudrate=9600)


# 测试
if __name__ == "__main__":
    # send_to_stm32(message="0")  # 发送 0x00
    # send_to_stm32(message="1")  # 发送 0x01
    pass

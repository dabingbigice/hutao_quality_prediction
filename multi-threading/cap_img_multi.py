def img_put_thread_start(queue_list_frame, queue_ist_ret, cap):
    print('开启多线程成功！！！！')
    while th_flag:
        start1 = time.time()
        # time.sleep(0.05)
        ret, frame = cap.read()
        start2 = time.time()
        try:
            queue_list_frame.put(frame, block=False)
            queue_ist_ret.put(ret, block=False)
        except queue.Full:
            print("队列已满，丢弃当前帧")
        print(f'{cap}:摄像头读取照片时间{(start2 - start1) * 1000}ms')
        print(f"Photo Added  by {threading.current_thread().name} ")

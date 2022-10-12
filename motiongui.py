from asyncio.windows_events import NULL
import tkinter
import tkinter.filedialog
import time
import datetime
from tkinter.font import Font

from gui_model_utils import *

# original version motion guide app
class MotionGuideGUI():
    def __init__(self, init_window_obj, active_duration = 2000, relax_duration = 1000):
        self.init_window_name = init_window_obj
        #self.progress_bar_len = 500
        # 控制小程序的运行和停止
        self.is_suspend = False
        # 用于循环的动作序列
        self.motion_sequence = ["收缩\n", "舒张\n"]
        self.motion_seq_len = len(self.motion_sequence)
        self.motion_index = 0
        # 倒计时
        self.count_down_sequence = ["5\n", "4\n", "3\n", "2\n", "1\n"]
        self.count_down_sequence_len = len(self.count_down_sequence)
        self.count_down_flag = False
        self.count_down_index = 0
        # 设置字体等其他参数
        self.myFont = Font(family="Times New Roman", size=12)
        self.motion_duration = active_duration
        self.relax_duration = relax_duration
        self.f_name = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())) + ".txt"

    def set_init_window(self):
        # 窗口名和尺寸设置
        self.init_window_name.title("motion guide GUI")#窗口名
        self.init_window_name.geometry('1000x500+300+500')#窗口尺寸和位置
        self.init_window_name.attributes("-alpha", 0.9)
        # label
        self.init_data_label = tkinter.Label(self.init_window_name, font = ('Arial', 20), text = "Follow The Instructions Bellow")
        self.init_data_label.grid(row = 0, column = 0)
        self.log_label = tkinter.Label(self.init_window_name, text = "Log Message")
        self.log_label.grid(row = 2, column = 0)
        # 文本框
        self.init_data_Text = tkinter.Text(self.init_window_name, font=('Arial', 20), width=30, height=2)  #原始数据录入框
        self.init_data_Text.grid(row=1, column=0, rowspan=1, columnspan=1)
        self.log_data_Text = tkinter.Text(self.init_window_name, width=50, height=20)  # 日志框
        self.log_data_Text.grid(row=3, column=0, rowspan=5, columnspan=1)
        self.db_show_text = tkinter.Text(self.init_window_name, width = 50, height = 2)
        self.db_show_text.grid(row = 2, column = 3, rowspan = 1, columnspan = 1)
        self.txt_show_text = tkinter.Text(self.init_window_name, width = 50, height = 2)
        self.txt_show_text.grid(row = 3, column = 3, rowspan = 1, columnspan = 1)
        self.model_show_text = tkinter.Text(self.init_window_name, width = 50, height = 2)
        self.model_show_text.grid(row = 4, column = 3, rowspan = 1, columnspan = 1)
        # 按键，控制开始和停止等功能
        self.start_button = tkinter.Button(self.init_window_name, text = "START", bg = "lightblue", width = 10, command = self.start)#调用内部方法，加()为直接调用
        self.start_button.grid(row = 1, column = 2)
        self.stop_button = tkinter.Button(self.init_window_name, text = "STOP", bg = "lightgreen", width = 10, command = self.stop)
        self.stop_button.grid(row = 1, column = 3)
        self.db_button = tkinter.Button(self.init_window_name, text = "Choose .db file", bg = "lightgreen", width = 20, command = self.choose_db_file)
        self.db_button.grid(row = 2, column = 2)
        self.txt_button = tkinter.Button(self.init_window_name, text = "Choose time file", bg = "lightgreen", width = 20, command = self.choose_timetxt_file)
        self.txt_button.grid(row = 3, column =  2)
        self.choose_model_button = tkinter.Button(self.init_window_name, text = "choose model file", bg = "lightgreem", width=20, command = self.choose_model)
        self.choose_model_button.grid(row = 4, column=2)
        self.analyze_button = tkinter.Button(self.init_window_name, text = "Analyze", bg = "lightgreen", width = 10, command = self.analyze)
        self.analyze_button.grid(row = 6, column = 2)

    def start(self):
        self.is_suspend = True
        self.count_down_flag = True
        pass

    def stop(self):
        self.is_suspend = False
        # self.count_down_flag = False
        # self.log_data_Text.insert(tkinter.END, "stop function here\r\n")
        pass

    def choose_db_file(self):
        self.db_file_path = tkinter.filedialog.askopenfilename()
        self.db_show_text.delete(1.0, tkinter.END)
        self.db_show_text.insert(1.0, self.db_file_path)
        pass

    def choose_timetxt_file(self):
        self.txt_file_path = tkinter.filedialog.askopenfilename()
        self.txt_show_text.delete(1.0, tkinter.END)
        self.txt_show_text.insert(1.0, self.txt_file_path)
        pass

    def choose_model(self):
        self.model_file_path = tkinter.filedialog.askopenfilename()
        self.model_show_text.delete(1.0, tkinter.END)
        self.model_show_text.insert(1.0, self.model_file_path)
        pass
    
    def analyze(self):
        # 实现分析功能
        pass

    def gui_loop(self):
        self.init_window_name.update()

        if self.is_suspend: # 程序开始
            if self.count_down_flag:    # 判断是否进入倒计时
                # 进入倒计时
                if self.count_down_index <= self.count_down_sequence_len - 1:
                    # 打印倒计时序列
                    self.init_data_Text.delete(1.0, tkinter.END)
                    self.init_data_Text.insert(1.0, self.count_down_sequence[self.count_down_index])
                    self.count_down_index += 1
                    self.init_window_name.after(1000, self.gui_loop)
                else:
                    # 倒计时完成一次后，flag改为false，只在程序开始时进入一次倒计时
                    self.count_down_flag = False
                    self.init_window_name.after(1000, self.gui_loop)
            else:
                # 不进入倒计时，打印动作提示信息
                if self.motion_index <= self.motion_seq_len-1:
                    # 打印动作提示
                    self.init_data_Text.delete(1.0, tkinter.END)
                    self.init_data_Text.insert(1.0, self.motion_sequence[self.motion_index])
                    # 
                    self.write_log_to_Text(self.motion_sequence[self.motion_index])
                    self.motion_index += 1
                    self.init_window_name.after(self.motion_duration, self.gui_loop)
                else:
                    self.motion_index = 0
                    self.init_window_name.after(self.relax_duration, self.gui_loop)
        else:   # 程序未开始运行
            self.init_window_name.after(0, self.gui_loop)
        #self.init_window_name.after(2000, self.gui_loop)
        pass
        
    def write_log_to_Text(self, logmsg):
        dt_s = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        dt_ms = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        logmsg_in = str(dt_ms) +" " + str(logmsg) + "\n"      #换行
        self.log_data_Text.insert(tkinter.END, logmsg_in)
        f = open(self.f_name, "a")
        f.write(logmsg_in)
        f.close()
        pass
    # end class
    pass


def gui_start():
    # 实例化父窗口
    init_window = tkinter.Tk()
    # 创建motion guide GUI类，设置窗口组间和属性
    win_a = MotionGuideGUI(init_window)
    win_a.set_init_window()
    # 运行gui_loop方法
    init_window.after(2000, win_a.gui_loop)
    # 父窗口进入事件循环，可以理解为保持窗口运行，否则界面不展示
    init_window.mainloop()
    pass


if __name__ == '__main__':
    gui_start()


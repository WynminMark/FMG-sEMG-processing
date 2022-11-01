"""
1.多路iFEMG信号选择，scaler和模型
2.针对病人的信号采集开始暂停功能
"""

import tkinter
import tkinter.filedialog
import time
import datetime
from tkinter.font import NORMAL, Font
from tkinter import DISABLED, ttk
# private 
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
        pass

    def set_init_window(self):
        # 窗口名和尺寸设置
        self.init_window_name.title("Motion Guide GUI")#窗口名
        self.init_window_name.geometry('1300x500+300+500')#窗口尺寸和位置
        self.init_window_name.attributes("-alpha", 0.9)
        # label
        self.init_data_label = tkinter.Label(self.init_window_name, font=('Arial', 20), text="Follow The Instructions Bellow")
        self.init_data_label.grid(row=0, column=0)
        self.log_label = tkinter.Label(self.init_window_name, text = "Log Message")
        self.log_label.grid(row=2, column=0)
        self.result_label = tkinter.Label(self.init_window_name, font=('Arial', 20), text="Result (kg): ")
        self.result_label.grid(row=0, column=5)
        self.subject_name_label = tkinter.Label(self.init_window_name, text="Subject Name: ")
        self.subject_name_label.grid(row=7, column=2)
        self.subject_level_label = tkinter.Label(self.init_window_name, text="Strength Level (kg): ")
        self.subject_level_label.grid(row=7, column=4)
        self.subject_height_label = tkinter.Label(self.init_window_name, text="Subject Height (cm): ")
        self.subject_height_label.grid(row=8, column=2)
        self.subject_weight_label = tkinter.Label(self.init_window_name, text="Subject Weight (kg): ")
        self.subject_weight_label.grid(row=8, column=4)
        self.subject_age_label = tkinter.Label(self.init_window_name, text="Subject Age: ")
        self.subject_age_label.grid(row=9, column=2)
        self.subject_gender_label = tkinter.Label(self.init_window_name, text="Subject Gender: ")
        self.subject_gender_label.grid(row=9, column=4)
        # 文本框
        self.init_data_Text = tkinter.Text(self.init_window_name, font=('Arial', 20), width=25, height=2)  #原始数据录入框
        self.init_data_Text.grid(row=1, column=0, rowspan=1, columnspan=1)
        self.log_data_Text = tkinter.Text(self.init_window_name, width=50, height=20)  # 日志框
        self.log_data_Text.grid(row=3, column=0, rowspan=5, columnspan=1)
        self.db_show_text = tkinter.Text(self.init_window_name, width=100, height=2, state=DISABLED)
        self.db_show_text.grid(row=2, column=3, rowspan=1, columnspan=3)
        self.txt_show_text = tkinter.Text(self.init_window_name, width=100, height=2, state=DISABLED)
        self.txt_show_text.grid(row=3, column=3, rowspan=1, columnspan=3)
        self.model_show_text = tkinter.Text(self.init_window_name, width=100, height=2, state=DISABLED)
        self.model_show_text.grid(row=4, column=3, rowspan=1, columnspan=3)
        self.result_show_text = tkinter.Text(self.init_window_name, font=('Arial', 20), width=20, height=2)
        self.result_show_text.grid(row=1, column=5)
        # 按键，控制开始和停止等功能
        self.start_button = tkinter.Button(self.init_window_name, text = "START", bg = "lightblue", width = 10, command = self.start)#调用内部方法，加()为直接调用
        self.start_button.grid(row = 1, column = 2)
        self.stop_button = tkinter.Button(self.init_window_name, text = "STOP", bg = "lightgreen", width = 10, command = self.stop)
        self.stop_button.grid(row = 1, column = 3)
        self.db_button = tkinter.Button(self.init_window_name, text = "Choose .db file", bg = "lightgreen", width = 20, command = self.choose_db_file)
        self.db_button.grid(row = 2, column = 2)
        self.txt_button = tkinter.Button(self.init_window_name, text = "Choose time file", bg = "lightgreen", width = 20, command = self.choose_timetxt_file)
        self.txt_button.grid(row = 3, column =  2)
        self.choose_model_button = tkinter.Button(self.init_window_name, text = "Choose model file", bg = "lightgreen", width=20, command = self.choose_model)
        self.choose_model_button.grid(row = 4, column=2)
        self.analyze_button = tkinter.Button(self.init_window_name, text = "Analyze", bg = "lightgreen", width = 10, command = self.analyze)
        self.analyze_button.grid(row = 1, column = 4)
        # channel多选框
        channel_var_1 = tkinter.IntVar()
        channel_var_2 = tkinter.IntVar()
        channel_var_3 = tkinter.IntVar()
        channel_var_4 = tkinter.IntVar()
        channel_var_5 = tkinter.IntVar()
        channel_var_6 = tkinter.IntVar()
        channel_var_7 = tkinter.IntVar()
        channel_var_8 = tkinter.IntVar()
        self.channel_var_list = [channel_var_1, channel_var_2, channel_var_3, channel_var_4, channel_var_5, channel_var_6, channel_var_7, channel_var_8]

        channel1_check_button = tkinter.Checkbutton(self.init_window_name, text="ch1", variable=channel_var_1, onvalue=1, offvalue=0)
        channel1_check_button.grid(row=5, column=2)
        channel2_check_button = tkinter.Checkbutton(self.init_window_name, text="ch2", variable=channel_var_2, onvalue=1, offvalue=0)
        channel2_check_button.grid(row=5, column=3)
        channel3_check_button = tkinter.Checkbutton(self.init_window_name, text="ch3", variable=channel_var_3, onvalue=1, offvalue=0)
        channel3_check_button.grid(row=5, column=4)
        channel4_check_button = tkinter.Checkbutton(self.init_window_name, text="ch4", variable=channel_var_4, onvalue=1, offvalue=0)
        channel4_check_button.grid(row=5, column=5)
        channel5_check_button = tkinter.Checkbutton(self.init_window_name, text="ch5", variable=channel_var_5, onvalue=1, offvalue=0)
        channel5_check_button.grid(row=6, column=2)
        channel6_check_button = tkinter.Checkbutton(self.init_window_name, text="ch6", variable=channel_var_6, onvalue=1, offvalue=0)
        channel6_check_button.grid(row=6, column=3)
        channel7_check_button = tkinter.Checkbutton(self.init_window_name, text="ch7", variable=channel_var_7, onvalue=1, offvalue=0)
        channel7_check_button.grid(row=6, column=4)
        channel8_check_button = tkinter.Checkbutton(self.init_window_name, text="ch8", variable=channel_var_8, onvalue=1, offvalue=0)
        channel8_check_button.grid(row=6, column=5)

        # gender下拉选择框
        self.gender_value = tkinter.StringVar()
        self.gender_value.set("Male")
        self.gender_combobox = ttk.Combobox(master=self.init_window_name, # 父容器
                                            height=10, # 高度,下拉显示的条目数量
                                            width=10, # 宽度
                                            state='readonly', # 设置状态 normal(可选可输入)、readonly(只可选)、 disabled
                                            cursor='arrow', # 鼠标移动时样式 arrow, circle, cross, plus...
                                            font=('', 20), # 字体
                                            textvariable=self.gender_value, # 通过StringVar设置可改变的值
                                            values=["Male", "Female"], # 设置下拉框的选项
                                            )
        self.gender_combobox.grid(row=9, column=5)
        # 输入框
        self.name_entry = tkinter.Entry(self.init_window_name)
        self.name_entry.grid(row=7, column=3)
        self.strength_level_entry = tkinter.Entry(self.init_window_name)
        self.strength_level_entry.grid(row=7, column=5)
        self.height_entry = tkinter.Entry(self.init_window_name)
        self.height_entry.grid(row=8, column=3)
        self.weight_entry = tkinter.Entry(self.init_window_name)
        self.weight_entry.grid(row=8, column=5)
        self.age_entry = tkinter.Entry(self.init_window_name)
        self.age_entry.grid(row=9, column=3)
        pass

    def start(self):
        self.is_suspend = True
        self.count_down_flag = True
        pass

    def stop(self):
        self.is_suspend = False
        pass

    def choose_db_file(self):
        self.db_file_path = tkinter.filedialog.askopenfilename()
        self.db_show_text.config(state=NORMAL)
        self.db_show_text.delete(1.0, tkinter.END)
        self.db_show_text.insert(1.0, self.db_file_path)
        self.db_show_text.config(state=DISABLED)
        pass

    def choose_timetxt_file(self):
        self.txt_file_path = tkinter.filedialog.askopenfilename()
        self.txt_show_text.config(state=NORMAL)
        self.txt_show_text.delete(1.0, tkinter.END)
        self.txt_show_text.insert(1.0, self.txt_file_path)
        self.txt_show_text.config(state=DISABLED)
        pass

    def choose_model(self):
        self.model_file_path = tkinter.filedialog.askopenfilename()
        self.model_show_text.config(state=NORMAL)
        self.model_show_text.delete(1.0, tkinter.END)
        self.model_show_text.insert(1.0, self.model_file_path)
        self.model_show_text.config(state=DISABLED)
        pass

    def get_channel_num(self):
        """
        get channel NO. from CheckButton
        """
        # get channel NO. from CheckButton
        # list: 1 for channel chosed and 0 for not
        channel_onoff = list(i.get() for i in self.channel_var_list)
        # get index
        channel_num = tuple(i for i, e in enumerate(channel_onoff) if e != 0)
        return channel_num
    
    def analyze(self):
        '''
        '''
        # 由于用到提示框提示错误信息，运行前清空
        self.init_data_Text.delete(1.0, tkinter.END)
        # 获得GUI界面输入
        try:
            gender_str = self.gender_combobox.get()
            name_str = self.name_entry.get()
            strength_level_float = float(self.strength_level_entry.get())
            height_float = float(self.height_entry.get())
            weight_float = float(self.weight_entry.get())
            age_float = float(self.age_entry.get())
        except ValueError:
            self.init_data_Text.delete(1.0, tkinter.END)
            self.init_data_Text.insert(1.0, "Check Input Information")
            return
            
        # 获得通道数输入
        signal_channel_num = self.get_channel_num()

        if gender_str == "Male":
            analysis_result = multi_channel_analysis(db_file_path = self.db_file_path,
                                                    time_file_path = self.txt_file_path,
                                                    model_file_path = self.model_file_path,
                                                    signal_channel = signal_channel_num,
                                                    subject_height = height_float,
                                                    subject_weight = weight_float,
                                                    subject_age = age_float,
                                                    subject_gender = 1,
                                                    subject_name = name_str,
                                                    strength_level = strength_level_float)  # demo中有时不需要输入肌力标签=np.NaN
        elif gender_str == "Female":
            analysis_result = multi_channel_analysis(db_file_path = self.db_file_path,
                                                    time_file_path = self.txt_file_path,
                                                    model_file_path = self.model_file_path,
                                                    signal_channel = signal_channel_num,
                                                    subject_height = height_float,
                                                    subject_weight = weight_float,
                                                    subject_age = age_float,
                                                    subject_gender = 0,
                                                    subject_name = name_str,
                                                    strength_level = strength_level_float)  # demo中有时不需要输入肌力标签=np.NaN
            pass
        self.result_show_text.delete(1.0, tkinter.END)
        self.result_show_text.insert(1.0, analysis_result)
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
        # dt_s = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
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


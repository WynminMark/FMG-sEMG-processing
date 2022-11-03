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
    def __init__(self, init_window_obj, active_duration = 4000, relax_duration = 5000):
        self.init_window_name = init_window_obj
        #self.progress_bar_len = 500
        # 控制小程序的运行和停止
        self.is_suspend = False
        # 用于循环的动作序列
        self.motion_sequence = ["收缩\n", "舒张\n"]
        self.motion_seq_len = len(self.motion_sequence)
        self.motion_index = 0
        # 倒计时
        self.count_down_sequence = ["3\n", "2\n", "1\n"]
        self.count_down_sequence_len = len(self.count_down_sequence)
        self.count_down_flag = False
        self.count_down_index = 0
        # 设置字体等其他参数
        self.myFont = Font(family="Times New Roman", size=12)
        self.motion_duration = active_duration
        self.relax_duration = relax_duration
        
        pass

    def set_init_window(self):
        # 窗口名和尺寸设置
        self.init_window_name.title("Motion Guide GUI")#窗口名
        self.init_window_name.geometry('1300x500+300+500')#窗口尺寸和位置
        self.init_window_name.attributes("-alpha", 0.9)
        # label
        self.init_data_label = tkinter.Label(self.init_window_name, font=('Arial', 20), text="Follow The Instructions Bellow")
        self.init_data_label.grid(row=0, column=0)
        self.log_label = tkinter.Label(self.init_window_name, font=('Arial', 10), text="Log Message")
        self.log_label.grid(row=2, column=0)
        self.result_label = tkinter.Label(self.init_window_name, font=('Arial', 20), text="Result (kg): ")
        self.result_label.grid(row=7, column=5)
        self.subject_name_label = tkinter.Label(self.init_window_name, text="Subject Name: ")
        self.subject_name_label.grid(row=0, column=5)
        self.subject_height_label = tkinter.Label(self.init_window_name, text="Subject Height (cm): ")
        self.subject_height_label.grid(row=1, column=5)
        self.subject_weight_label = tkinter.Label(self.init_window_name, text="Subject Weight (kg): ")
        self.subject_weight_label.grid(row=2, column=5)
        self.subject_age_label = tkinter.Label(self.init_window_name, text="Subject Age: ")
        self.subject_age_label.grid(row=3, column=5)
        self.subject_gender_label = tkinter.Label(self.init_window_name, text="Subject Gender: ")
        self.subject_gender_label.grid(row=4, column=5)
        self.subject_level_label = tkinter.Label(self.init_window_name, text="Strength Level (kg): ")
        self.subject_level_label.grid(row=5, column=5)
        self.agonist_ch1_label = tkinter.Label(self.init_window_name, text="Agonist Ch1")
        self.agonist_ch1_label.grid(row=4, column=1)
        self.agonist_ch2_label = tkinter.Label(self.init_window_name, text="Agonist Ch2")
        self.agonist_ch2_label.grid(row=5, column=1)
        self.agonist_ch3_label = tkinter.Label(self.init_window_name, text="Agonist Ch3")
        self.agonist_ch3_label.grid(row=6, column=1)
        self.agonist_ch4_label = tkinter.Label(self.init_window_name, text="Agonist Ch4")
        self.agonist_ch4_label.grid(row=7, column=1)
        self.antagonist_ch1_label = tkinter.Label(self.init_window_name, text="Antagonist Ch1")
        self.antagonist_ch1_label.grid(row=4, column=3)
        self.antagonist_ch2_label = tkinter.Label(self.init_window_name, text="Antagonist Ch2")
        self.antagonist_ch2_label.grid(row=5, column=3)
        self.antagonist_ch3_label = tkinter.Label(self.init_window_name, text="Antagonist Ch3")
        self.antagonist_ch3_label.grid(row=6, column=3)
        self.antagonist_ch4_label = tkinter.Label(self.init_window_name, text="Antagonist Ch4")
        self.antagonist_ch4_label.grid(row=7, column=3)
        # 文本框
        self.init_data_Text = tkinter.Text(self.init_window_name, font=('Arial', 20), width=25, height=2)  #原始数据录入框
        self.init_data_Text.grid(row=1, column=0, rowspan=1, columnspan=1)
        self.log_data_Text = tkinter.Text(self.init_window_name, width=50, height=20)  # 日志框
        self.log_data_Text.grid(row=3, column=0, rowspan=5, columnspan=1)
        self.db_show_text = tkinter.Text(self.init_window_name, width=50, height=2, state=DISABLED)
        self.db_show_text.grid(row=1, column=3, rowspan=1, columnspan=2)
        self.txt_show_text = tkinter.Text(self.init_window_name, width=50, height=2, state=DISABLED)
        self.txt_show_text.grid(row=2, column=3, rowspan=1, columnspan=2)
        self.model_show_text = tkinter.Text(self.init_window_name, width=50, height=2, state=DISABLED)
        self.model_show_text.grid(row=3, column=3, rowspan=1, columnspan=2)
        self.result_show_text = tkinter.Text(self.init_window_name, font=('Arial', 10), width=20, height=2)
        self.result_show_text.grid(row=7, column=6)
        # 按键，控制开始和停止等功能
        self.start_button = tkinter.Button(self.init_window_name, text = "START", bg = "lightblue", width = 10, command = self.start)#调用内部方法，加()为直接调用
        self.start_button.grid(row=0, column=1, columnspan=2)
        self.stop_button = tkinter.Button(self.init_window_name, text = "STOP", bg = "lightgreen", width = 10, command = self.stop)
        self.stop_button.grid(row=0, column=3, columnspan=2)
        self.db_button = tkinter.Button(self.init_window_name, text = "Choose .db file", bg = "lightgreen", width = 20, command = self.choose_db_file)
        self.db_button.grid(row=1, column=1, columnspan=2)
        self.txt_button = tkinter.Button(self.init_window_name, text = "Choose time file", bg = "lightgreen", width = 20, command = self.choose_timetxt_file)
        self.txt_button.grid(row=2, column=1, columnspan=2)
        self.choose_model_button = tkinter.Button(self.init_window_name, text = "Choose model file", bg = "lightgreen", width=20, command = self.choose_model)
        self.choose_model_button.grid(row=3, column=1, columnspan=2)
        self.analyze_button = tkinter.Button(self.init_window_name, text = "Analyze", bg = "lightgreen", width = 10, command = self.get_channel_num)
        self.analyze_button.grid(row=6, column=5, columnspan=2)
        # channel choosing Combobox
        ch1_combobox_value = tkinter.IntVar()
        ch2_combobox_value = tkinter.IntVar()
        ch3_combobox_value = tkinter.IntVar()
        ch4_combobox_value = tkinter.IntVar()
        ch5_combobox_value = tkinter.IntVar()
        ch6_combobox_value = tkinter.IntVar()
        ch7_combobox_value = tkinter.IntVar()
        ch8_combobox_value = tkinter.IntVar()
        self.channel_var_list = [ch1_combobox_value, ch2_combobox_value, ch3_combobox_value, ch4_combobox_value, ch5_combobox_value, ch6_combobox_value, ch7_combobox_value, ch8_combobox_value]
        # set init value as 0
        for ch in self.channel_var_list:
            ch.set(0)
            pass
        ch_value_list = [0,1,2,3,4,5,6,7,8]
        self.agonist_ch1_combobox = ttk.Combobox(master=self.init_window_name, state="readonly", textvariable=ch1_combobox_value, values=ch_value_list)
        self.agonist_ch1_combobox.grid(row=4, column=2)
        self.agonist_ch2_combobox = ttk.Combobox(master=self.init_window_name, state="readonly", textvariable=ch2_combobox_value, values=ch_value_list)
        self.agonist_ch2_combobox.grid(row=5, column=2)
        self.agonist_ch3_combobox = ttk.Combobox(master=self.init_window_name, state="readonly", textvariable=ch3_combobox_value, values=ch_value_list)
        self.agonist_ch3_combobox.grid(row=6, column=2)
        self.agonist_ch4_combobox = ttk.Combobox(master=self.init_window_name, state="readonly", textvariable=ch4_combobox_value, values=ch_value_list)
        self.agonist_ch4_combobox.grid(row=7, column=2)
        self.antagonist_ch1_combobox = ttk.Combobox(master=self.init_window_name, state="readonly", textvariable=ch5_combobox_value, values=ch_value_list)
        self.antagonist_ch1_combobox.grid(row=4, column=4)
        self.antagonist_ch2_combobox = ttk.Combobox(master=self.init_window_name, state="readonly", textvariable=ch6_combobox_value, values=ch_value_list)
        self.antagonist_ch2_combobox.grid(row=5, column=4)
        self.antagonist_ch3_combobox = ttk.Combobox(master=self.init_window_name, state="readonly", textvariable=ch7_combobox_value, values=ch_value_list)
        self.antagonist_ch3_combobox.grid(row=6, column=4)
        self.antagonist_ch4_combobox = ttk.Combobox(master=self.init_window_name, state="readonly", textvariable=ch8_combobox_value, values=ch_value_list)
        self.antagonist_ch4_combobox.grid(row=7, column=4)
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
        self.gender_combobox.grid(row=4, column=6)
        # 输入框
        self.name_entry = tkinter.Entry(self.init_window_name)
        self.name_entry.grid(row=0, column=6)
        self.height_entry = tkinter.Entry(self.init_window_name)
        self.height_entry.grid(row=1, column=6)
        self.weight_entry = tkinter.Entry(self.init_window_name)
        self.weight_entry.grid(row=2, column=6)
        self.age_entry = tkinter.Entry(self.init_window_name)
        self.age_entry.grid(row=3, column=6)
        self.strength_level_entry = tkinter.Entry(self.init_window_name)
        self.strength_level_entry.grid(row=5, column=6)
        pass

    def start(self):
        # start 后重新建立新文件
        self.f_name = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())) + ".txt"
        # 开始进入loop循环
        self.is_suspend = True
        # 开始新一轮倒计时
        self.count_down_flag = True
        self.count_down_index = 0
        # 清空上一轮log
        self.log_data_Text.delete(1.0, tkinter.END)
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
        print("channel ", channel_onoff)
        print(channel_num)
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
                    # 倒计时完成一次后，flag改为false，进入动作循环
                    self.count_down_flag = False
                    # 设置1000倒计时结束后2s后开始动作循环
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
        with open(self.f_name, "a") as f:
            f.write(logmsg_in)
        pass
    # end class
    pass


def gui_start():
    # 实例化父窗口
    init_window = tkinter.Tk()
    # 创建motion guide GUI类，设置窗口组间和属性
    win_a = MotionGuideGUI(init_window, active_duration=4000, relax_duration=0)
    win_a.set_init_window()
    # 运行gui_loop方法
    init_window.after(2000, win_a.gui_loop)
    # 父窗口进入事件循环，可以理解为保持窗口运行，否则界面不展示
    init_window.mainloop()
    pass


if __name__ == '__main__':
    gui_start()


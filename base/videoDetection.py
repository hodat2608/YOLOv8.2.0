from base_model import Base,MySQL_Connection,PLC_Connection,removefile,base_handle_video
import sys
import os
current_dir = os.getcwd()
sys.path.append(str(current_dir))
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import torch
import os
from PIL import Image,ImageTk
from tkinter import ttk
import tkinter as tk
import sys
import os
from base_model import Base

class Video_Dectection(Base,base_handle_video): 

    def __init__(self, *args, **kwargs):
        super(Video_Dectection, self).__init__(*args, **kwargs)
        super(Base).__init__()
        super(base_handle_video).__init__()
        torch.cuda.set_device(0)
        self.database = MySQL_Connection("127.0.0.1","root1","987654321","connect_database_model") 
        self.name_table = 'model_connection_model_VID'
        self.item_code_cfg = "EDFWTA"
        self.image_files = []
        self.current_image_index = -1
        self.state = 1
        self.password = "123"
        self.lockable_widgets = [] 
        self.lock_params = []
        self.model_name_labels = []
        self.join = []
        self.ok_vars = []
        self.ng_vars = []
        self.num_inputs = []
        self.wn_inputs = []
        self.wx_inputs = []
        self.hn_inputs = []
        self.hx_inputs = []
        self.plc_inputs = []
        self.conf_scales = []
        self.widgets_option_layout_parameters = []
        self.row_widgets = []
        self.weights = None
        self.scale_conf_all = None
        self.size_model = None
        self.item_code = []
        self.make_cls_var = False
        self.permisson_btn = []
        self.model = None
        self.time_processing_output = None
        self.result_detection = None
        self.cls = False
        self.progress_label = None
        self.path_video = None
        self.video_canvas = None
        self.device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_data_model(self):
        return super().load_data_model_vid()

    def load_parameters_model(self, model1, load_path_weight, load_item_code, load_confidence_all_scale, records):
        return super().load_parameters_model_vid(model1, load_path_weight, load_item_code, load_confidence_all_scale, records)
    
    def toggle_state_option_layout_parameters(self):
        return super().toggle_state_option_layout_parameters()

    def select_video(self, path_video):
        return super().select_video(path_video)
    
    def start_rendering(self,path_video, progress_label, video_canvas):
        return super().start_rendering(path_video, progress_label, video_canvas)
    
    def Video_Settings(self,settings_notebook):

        records,load_path_weight,load_item_code,load_confidence_all_scale = self.load_data_model()
        if load_path_weight:
            self.model = YOLO(load_path_weight, task='detect').to(device=self.device1)
        else: 
            load_path_weight = 'yolov8n.pt'
            self.model = YOLO(load_path_weight, task='detect').to(device=self.device1)
        filename =r"C:\Users\CCSX009\Documents\ultralytics-main\2024-03-05_00-01-31-398585-C1.jpg"
        self.model(filename,imgsz=608,conf=0.2)
        print('Load model 1 successfully')
        camera_settings_tab = ttk.Frame(settings_notebook)
        settings_notebook.add(camera_settings_tab, text="Camera 1")

        canvas1 = tk.Canvas(camera_settings_tab)
        scrollbar = ttk.Scrollbar(camera_settings_tab, orient="vertical", command=canvas1.yview)
        scrollable_frame = ttk.Frame(canvas1)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas1.configure(
                scrollregion=canvas1.bbox("all")
            )
        )
        canvas1.bind_all("<MouseWheel>", lambda event: canvas1.yview_scroll(int(-1*(event.delta/120)), "units"))
        canvas1.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas1.configure(yscrollcommand=scrollbar.set)

        canvas1.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        camera_settings_tab.grid_columnconfigure(0, weight=1)
        camera_settings_tab.grid_rowconfigure(0, weight=1)

        frame_width = 1500
        frame_height = 2000

        Frame_1 = ttk.LabelFrame(scrollable_frame, text="Frame 1", width=frame_width, height=frame_height)
        Frame_2 = ttk.LabelFrame(scrollable_frame, text="Frame 2", width=frame_width, height=frame_height)

        Frame_1.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")  
        Frame_2.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
           
        scrollable_frame.grid_columnconfigure(0, weight=1)
        scrollable_frame.grid_columnconfigure(1, weight=1)
        scrollable_frame.grid_rowconfigure(0, weight=1)
        scrollable_frame.grid_rowconfigure(1, weight=1)

        self.option_layout_models(Frame_1,Frame_2,records)
        self.option_layout_parameters(Frame_2,self.model)
        self.load_parameters_model(self.model,load_path_weight,load_item_code,load_confidence_all_scale,records)
        self.toggle_state_option_layout_parameters()

    def option_layout_models(self, Frame_1, Frame_2,records):

        ttk.Label(Frame_1, text='1. File train detect model', font=('Segoe UI', 12)).grid(column=0, row=0, padx=10, pady=5, sticky="nws")

        self.weights = ttk.Entry(Frame_1, width=60)
        self.weights.grid(row=1, column=0, columnspan=5, padx=(30, 5), pady=5, sticky="w", ipadx=20, ipady=2)

        button_frame = ttk.Frame(Frame_1)
        button_frame.grid(row=2, column=0, columnspan=2, padx=(30, 30), pady=5, sticky="w")

        change_model_button = tk.Button(button_frame, text="Change Model", command=lambda: self.change_model(Frame_2))
        change_model_button.grid(row=0, column=0, padx=(0, 8), pady=5, sticky="w", ipadx=5, ipady=2)
        change_model_button.config(state="disabled")
        self.lockable_widgets.append(change_model_button)

        load_parameters = tk.Button(button_frame, text="Load Parameters", command=lambda: self.load_parameters_from_weight(records))
        load_parameters.grid(row=0, column=1, padx=(0, 8), pady=5, sticky="w", ipadx=5, ipady=2)
        load_parameters.config(state="disabled")
        self.lockable_widgets.append(load_parameters)

        custom_para = tk.Button(button_frame, text="Custom Parameters")
        custom_para.grid(row=0, column=2, padx=(0, 8), pady=5, sticky="w", ipadx=5, ipady=2)
        custom_para.config(state="disabled")
        self.lockable_widgets.append(custom_para)

        self.permisson_btn = tk.Button(button_frame, text="Unlock", command=lambda: self.toggle_state_layout_model())
        self.permisson_btn.grid(row=0, column=3, padx=(0, 8), pady=5, sticky="w", ipadx=5, ipady=2)

        label_scale_conf_all = ttk.Label(Frame_1, text='2. Confidence Threshold', font=('Segoe UI', 12))
        label_scale_conf_all.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="nws")

        self.scale_conf_all = tk.Scale(Frame_1, from_=1, to=100, orient='horizontal', length=400)
        self.scale_conf_all.grid(row=4, column=0, columnspan=2, padx=30, pady=5, sticky="nws")
        self.lockable_widgets.append(self.scale_conf_all)
        
        label_size_model = ttk.Label(Frame_1, text='2. Size Model', font=('Segoe UI', 12))
        label_size_model.grid(row=5, column=0, columnspan=2, padx=10, pady=5, sticky="nws")

        options = [468, 608, 768]
        self.size_model = ttk.Combobox(Frame_1, values=options, width=7)
        self.size_model.grid(row=6, column=0, columnspan=2, padx=30, pady=5, sticky="nws", ipadx=5, ipady=2)
        self.size_model.set(608)
        self.lockable_widgets.append(self.size_model)

        name_item_code = ttk.Label(Frame_1, text='3. Item Code', font=('Segoe UI', 12))
        name_item_code.grid(row=7, column=0, columnspan=2, padx=10, pady=5, sticky="nws")

        self.item_code = ttk.Entry(Frame_1, width=10)
        self.item_code.grid(row=8, column=0, columnspan=2, padx=30, pady=5, sticky="w", ipadx=5, ipady=2)
        self.lockable_widgets.append(self.item_code)
      
        save_data_to_database = ttk.Button(Frame_1, text='Apply', command=lambda: self.save_params_model())
        save_data_to_database.grid(row=9, column=0, columnspan=2, padx=30, pady=5, sticky="w", ipadx=5, ipady=2)
        save_data_to_database.config(state="disabled")
        self.lockable_widgets.append(save_data_to_database)

        video_frame_display = ttk.Label(Frame_1, text='4. Video Processing', font=('Segoe UI', 12))
        video_frame_display.grid(row=10, column=0, columnspan=2, padx=10, pady=5, sticky="nws")

        video_canvas = tk.Canvas(Frame_1, width=600, height=550, bg="black")
        video_canvas.grid(row=11, column=0, columnspan=2, padx=30, pady=5, sticky="nws")

        path_video = ttk.Entry(Frame_1, width=55)
        path_video.grid(row=12, column=0, columnspan=2, padx=30, pady=5, sticky="w", ipadx=20, ipady=2)

        render_bttn_frame = ttk.Frame(Frame_1)
        render_bttn_frame.grid(row=12, column=1, columnspan=2, padx=0, pady=5, sticky="w")

        choose_path_video = tk.Button(render_bttn_frame, text="...", command=lambda: self.select_video(path_video))
        choose_path_video.grid(row=0, column=0, padx=0, pady=5, sticky="w", ipadx=5, ipady=2)
        choose_path_video.config(state="disabled")
        self.lockable_widgets.append(choose_path_video)

        render_button = tk.Button(render_bttn_frame, text="Start Render", command=lambda: self.start_rendering(path_video=path_video.get(),progress_label=progress_label,video_canvas=video_canvas))
        render_button.grid(row=0, column=1, padx=5,pady=5, sticky="w", ipadx=5, ipady=2)
        render_button.config(state="disabled")
        self.lockable_widgets.append(render_button)

        progress_label = ttk.Label(Frame_1, text="", font=('Segoe UI', 12))
        progress_label.grid(row=13, column=0, columnspan=2, padx=10, pady=5, sticky="nws")
    
    def option_layout_parameters(self,Frame_2,model):
        
        def ng_selected(row_widgets):
            ok_checkbox_var = row_widgets[2].var
            ng_checkbox_var = row_widgets[3].var
            if ng_checkbox_var.get() == True:
                ok_checkbox_var.set(False)

        def ok_selected(row_widgets):
            ok_checkbox_var = row_widgets[2].var
            ng_checkbox_var = row_widgets[3].var
            if ok_checkbox_var.get() == True:
                ng_checkbox_var.set(False)
    
        label = tk.Label(Frame_2, text='LABEL', fg='red', font=('Ubuntu', 12), width=12, anchor='center', relief="groove", borderwidth=2)
        label.grid(row=0, column=0, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

        joint_detect = tk.Label(Frame_2, text='join', fg='red', font=('Ubuntu', 12), anchor='center', relief="groove", borderwidth=2)
        joint_detect.grid(row=0, column=1, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

        ok_joint = tk.Label(Frame_2, text='OK', fg='red', font=('Ubuntu', 12), anchor='center', relief="groove", borderwidth=2)
        ok_joint.grid(row=0, column=2, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

        ng_joint = tk.Label(Frame_2, text='NG', fg='red', font=('Ubuntu', 12), anchor='center', relief="groove", borderwidth=2)
        ng_joint.grid(row=0, column=3, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

        num_lb = tk.Label(Frame_2, text='NUM', fg='red', font=('Ubuntu', 12), width=7, anchor='center', relief="groove", borderwidth=2)
        num_lb.grid(row=0, column=4, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

        width_n = tk.Label(Frame_2, text='W_N', fg='red', font=('Ubuntu', 12), width=7, anchor='center', relief="groove", borderwidth=2)
        width_n.grid(row=0, column=5, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

        wight_x= tk.Label(Frame_2, text='W_X', fg='red', font=('Ubuntu', 12), width=7, anchor='center', relief="groove", borderwidth=2)
        wight_x.grid(row=0, column=6, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

        height_n = tk.Label(Frame_2, text='H_N', fg='red', font=('Ubuntu', 12), width=7, anchor='center', relief="groove", borderwidth=2)
        height_n.grid(row=0, column=7, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

        height_x = tk.Label(Frame_2, text='H_X', fg='red', font=('Ubuntu', 12), width=7, anchor='center', relief="groove", borderwidth=2)
        height_x.grid(row=0, column=8, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

        plc_var = tk.Label(Frame_2, text='PLC', fg='red', font=('Ubuntu', 12), width=7, anchor='center', relief="groove", borderwidth=2)
        plc_var.grid(row=0, column=9, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

        conf = tk.Label(Frame_2, text='CONFIDENCE THRESHOLD', fg='red', font=('Ubuntu', 12), width=25, anchor='center', relief="groove", borderwidth=2)
        conf.grid(row=0, column=10, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

        widgets_option_layout_parameters = []

        self.model_name_labels.clear()
        self.join.clear()
        self.ok_vars.clear()
        self.ng_vars.clear()
        self.num_inputs.clear()
        self.wn_inputs.clear()
        self.wx_inputs.clear()
        self.hn_inputs.clear()
        self.hx_inputs.clear()
        self.plc_inputs.clear()
        self.conf_scales.clear()

        for i1 in range(len(model.names)):
            row_widgets = []

            model_name_label = tk.Label(Frame_2, text=f'{model.names[i1]}', fg='black', font=('Segoe UI', 12), width=15, anchor='w')
            row_widgets.append(model_name_label)
            self.model_name_labels.append(model_name_label)

            join_checkbox_var = tk.BooleanVar()
            join_checkbox = tk.Checkbutton(Frame_2, variable=join_checkbox_var, onvalue=True, offvalue=False,anchor='w')
            join_checkbox.grid()
            join_checkbox.var = join_checkbox_var
            row_widgets.append(join_checkbox)
            self.join.append(join_checkbox_var)
            self.lock_params.append(join_checkbox)
            self.lockable_widgets.append(join_checkbox)

            ok_checkbox_var = tk.BooleanVar()
            ok_checkbox = tk.Checkbutton(Frame_2, variable=ok_checkbox_var, onvalue=True, offvalue=False, anchor='w')
            ok_checkbox.grid()
            ok_checkbox.var = ok_checkbox_var
            row_widgets.append(ok_checkbox)
            self.ok_vars.append(ok_checkbox_var)
            self.lock_params.append(ok_checkbox)
            self.lockable_widgets.append(ok_checkbox)

            ng_checkbox_var = tk.BooleanVar()
            ng_checkbox = tk.Checkbutton(Frame_2, variable=ng_checkbox_var, onvalue=True, offvalue=False, anchor='w')
            ng_checkbox.grid()
            ng_checkbox.var = ng_checkbox_var
            row_widgets.append(ng_checkbox)
            self.ng_vars.append(ng_checkbox_var)
            self.lock_params.append(ng_checkbox)
            self.lockable_widgets.append(ng_checkbox)

            num_input = tk.Entry(Frame_2, width=7,)
            num_input.insert(0, '1')
            row_widgets.append(num_input)
            self.num_inputs.append(num_input)
            self.lock_params.append(num_input)
            self.lockable_widgets.append(num_input)

            wn_input = tk.Entry(Frame_2, width=7, )
            wn_input.insert(0, '0')
            row_widgets.append(wn_input)
            self.wn_inputs.append(wn_input)
            self.lock_params.append(wn_input)
            self.lockable_widgets.append(wn_input)

            wx_input = tk.Entry(Frame_2, width=7, )
            wx_input.insert(0, '1600')
            row_widgets.append(wx_input)
            self.wx_inputs.append(wx_input)
            self.lock_params.append(wx_input)
            self.lockable_widgets.append(wx_input)

            hn_input = tk.Entry(Frame_2, width=7, )
            hn_input.insert(0, '0')
            row_widgets.append(hn_input)
            self.hn_inputs.append(hn_input)
            self.lock_params.append(hn_input)
            self.lockable_widgets.append(hn_input)

            hx_input = tk.Entry(Frame_2, width=7, )
            hx_input.insert(0, '1200')
            row_widgets.append(hx_input)
            self.hx_inputs.append(hx_input)
            self.lock_params.append(hx_input)
            self.lockable_widgets.append(hx_input)

            plc_input = tk.Entry(Frame_2, width=7,)
            plc_input.insert(0, '0')
            row_widgets.append(plc_input)
            self.plc_inputs.append(plc_input)
            self.lock_params.append(plc_input)
            self.lockable_widgets.append(plc_input)

            conf_scale = tk.Scale(Frame_2, from_=1, to=100, orient='horizontal', length=250)
            row_widgets.append(conf_scale)
            self.conf_scales.append(conf_scale)
            self.lock_params.append(conf_scale)
            self.lockable_widgets.append(conf_scale)

            widgets_option_layout_parameters.append(row_widgets)

            for i, row in enumerate(widgets_option_layout_parameters):
                for j, widget in enumerate(row):
                    widget.grid(row=i+1, column=j, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)


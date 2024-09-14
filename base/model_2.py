from base_model import Base,MySQL_Connection,PLC_Connection,removefile
import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent.parent
ultralytics_main_dir = current_dir
sys.path.append(str(ultralytics_main_dir))
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import glob
# import stapipy as st
import cv2
from PIL import Image,ImageTk
import time
from tkinter import ttk
import tkinter as tk
import shutil
import os


class Model_Camera_2(Base,MySQL_Connection,PLC_Connection):

    def __init__(self, *args, **kwargs):
        super(Model_Camera_2, self).__init__(*args, **kwargs)
        super().__init__()
        self.database = MySQL_Connection("127.0.0.1","root1","987654321","connect_database_model") 
        self.name_table = 'model_connection_model2'
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
        self.weights = []
        self.scale_conf_all = None
        self.size_model = None
        self.item_code = []
        self.make_cls_var = []
        self.permisson_btn = []
        self.model = None
        self.time_processing_output = None
        self.result_detection = None
    
    def mohica(self):
        filepath= f"C:/Users/CCSX009/Documents/yolov5/test_image/camera2"
        filename = "C:/FH/New folder (3)/2024-05-21_15-50-51-289134_luu_xuat.jpg"
        shutil.copy(filename,filepath)

    def display_images_c2(self, camera_frame, camera_number):
        width = 800
        height = 800
        t1 = time.time()
        image_paths = glob.glob(f"C:/Users/CCSX009/Documents/yolov5/test_image/camera{camera_number}/*.jpg")
        if len(image_paths) == 0:
            pass
        else:
            for filename in image_paths:
                img1_orgin = cv2.imread(filename)
                for widget in camera_frame.winfo_children():
                    widget.destroy()
                image_result,results_detect,list_label_ng = self.processing_handle_image_customize(img1_orgin,width,height)
                t2 = time.time() - t1
                time_processing = str(int(t2*1000)) + 'ms'
                self.time_processing_output.config(text=f'{time_processing}')
                if results_detect == 'OK':
                    self.result_detection.config(text=results_detect,fg='green')
                else:
                    self.result_detection.config(text=results_detect,fg='red')
                list_label_ng = ','.join(list_label_ng)
                img_pil = Image.fromarray(image_result)
                photo = ImageTk.PhotoImage(img_pil)
                canvas = tk.Canvas(camera_frame, width=width, height=height)
                canvas.grid(row=2, column=0, padx=10, pady=10, sticky='ew')
                canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                canvas.image = photo
                canvas.create_text(10, 10, anchor=tk.NW, text=f'Time: {time_processing}', fill='black', font=('Segoe UI', 20))
                canvas.create_text(10, 40, anchor=tk.NW, text=f'Result: {results_detect}', fill='green' if results_detect == 'OK' else 'red', font=('Segoe UI', 20))
                canvas.create_text(10, 70, anchor=tk.NW, text=f'Label: {list_label_ng}', fill='red', font=('Segoe UI', 20))
                os.remove(filename)

    def update_images(self,window, camera2_frame):
        self.display_images_c2(camera2_frame,2)
        window.after(50, self.update_images,window,camera2_frame)
        
    def create_camera_frame_cam2(self, tab1, camera_number):

        style = ttk.Style()
        style.configure("Custom.TLabelframe", borderwidth=0)
        style.configure("Custom.TLabelframe.Label", background="white", foreground="white")

        frame_2 = ttk.LabelFrame(tab1, width=900, height=900, style="Custom.TLabelframe")
        frame_2.grid(row=0, column=1, padx=200, pady=10, sticky="nsew")

        camera_frame = ttk.LabelFrame(frame_2, text=f"Camera {camera_number}", width=800, height=800)
        camera_frame.grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

        time_frame = ttk.LabelFrame(frame_2, text=f"Time Processing Camera {camera_number}", width=300, height=100)
        time_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        self.time_processing_output = tk.Label(time_frame, text='0 ms', fg='black', font=('Segoe UI', 30), width=10, height=1, anchor='center')
        self.time_processing_output.grid(row=0, column=0, padx=10, pady=10, sticky='ew')

        result = ttk.LabelFrame(frame_2, text=f"Result Camera {camera_number}", width=300, height=100)
        result.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        self.result_detection = tk.Label(result, text='ERROR', fg='red', font=('Segoe UI', 30), width=10, height=1, anchor='center')
        self.result_detection.grid(row=0, column=0, padx=10, pady=10, sticky='ew')

        bonus = ttk.LabelFrame(frame_2, text=f"Bonus {camera_number}", width=300, height=100)
        bonus.grid(row=1, column=2, padx=10, pady=5, sticky="ew")

        bonus_test = tk.Label(bonus, text='Bonus', fg='red', font=('Segoe UI', 30), width=10, height=1, anchor='center')
        bonus_test.grid(row=0, column=0, padx=10, pady=10, sticky='ew')

        frame_2.grid_columnconfigure(0, weight=1)
        frame_2.grid_columnconfigure(1, weight=1)
        frame_2.grid_columnconfigure(2, weight=1)
        frame_2.grid_rowconfigure(0, weight=1)
        frame_2.grid_rowconfigure(1, weight=1)

        return camera_frame

    def Display_Camera(self,tab2):
        camera2_frame = self.create_camera_frame_cam2(tab2, 2)   
        return camera2_frame
    
    def connect_database(self):
        cursor,db_connection,check_connection,reconnect = super().connect_database()
        return cursor,db_connection,check_connection,reconnect
    
    def save_params_model(self):
        return super().save_params_model()
    
    def processing_handle_image_customize(self, input_image, width, height):
        return super().processing_handle_image_customize(input_image, width, height)
    
    # def load_data_model(self):
    #     return super().load_data_model()
    
    def load_parameters_model(self, model1, load_path_weight, load_item_code, load_confidence_all_scale, records):
        return super().load_parameters_model(model1, load_path_weight, load_item_code, load_confidence_all_scale, records)
    
    def change_model(self, Frame_2):
        return super().change_model(Frame_2)
    
    def load_params_child(self):
        return super().load_params_child()
    
    def load_parameters_from_weight(self, records):
        return super().load_parameters_from_weight(records)
    
    def handle_image(self, img1_orgin, width, height, camera_frame):
        return super().handle_image(img1_orgin, width, height, camera_frame)
    
    def detect_single_img(self, camera_frame):
        return super().detect_single_img(camera_frame)
    
    def detect_multi_img(self, camera_frame):
        return super().detect_multi_img(camera_frame)
    
    def show_image(self, index, camera_frame):
        return super().show_image(index, camera_frame)
    
    def detect_next_img(self, camera_frame):
        return super().detect_next_img(camera_frame)
    
    def detect_previos_img(self, camera_frame):
        return super().detect_previos_img(camera_frame)
    
    def detect_auto(self, camera_frame):
        return super().detect_auto(camera_frame) 
    
    def logging(self, folder_ok, folder_ng, logging_ok_checkbox_var, logging_ng_checkbox_var, camera_frame, percent_entry, logging_frame):
        return super().logging(folder_ok, folder_ng, logging_ok_checkbox_var, logging_ng_checkbox_var, camera_frame, percent_entry, logging_frame)
    
    def toggle_state_layout_model(self):
        return super().toggle_state_layout_model()
    
    def toggle_widgets_state(self, state):
        return super().toggle_widgets_state(state)
    
    def toggle_state_option_layout_parameters(self):
        return super().toggle_state_option_layout_parameters()
    
    def pick_folder_ng(self, folder_ng):
        return super().pick_folder_ng(folder_ng)
    
    def pick_folder_ok(self, folder_ok):
        return super().pick_folder_ok(folder_ok)
    
    def _make_cls(self, image_path_mks_cls, results, model_settings):
        return super()._make_cls(image_path_mks_cls, results, model_settings)
    
    def classify_imgs(self):
        return super().classify_imgs()
    
    def processing_handle_image_local(self, input_image_original, width, height, cls=False):
        return super().processing_handle_image_local(input_image_original, width, height, cls)
    
    def Camera_Settings(self,settings_notebook):
        # try: 
        records,load_path_weight,load_item_code,load_confidence_all_scale = self.load_data_model()
        # except : 
        #     records,load_path_weight = None,r"C:\Users\CCSX009\Documents\ultralytics-main\best.pt"
        self.model = YOLO(load_path_weight)
        filename =r"C:\Users\CCSX009\Documents\ultralytics-main\2024-03-05_00-01-31-398585-C1.jpg"
        self.model(filename,imgsz=608,conf=0.2)
        print('Load model 2 successfully')
        camera_settings_tab = ttk.Frame(settings_notebook)
        settings_notebook.add(camera_settings_tab, text="Camera 2")

        canvas = tk.Canvas(camera_settings_tab)
        scrollbar = ttk.Scrollbar(camera_settings_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        canvas.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(int(-1*(event.delta/120)), "units"))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky="nsew")
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

        options = [480, 608, 832]
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

        camera_frame_display = ttk.Label(Frame_1, text='4. Modify Image', font=('Segoe UI', 12))
        camera_frame_display.grid(row=10, column=0, columnspan=2, padx=10, pady=5, sticky="nws")

        camera_frame = ttk.LabelFrame(Frame_1, text=f"Camera 1", width=500, height=500)
        camera_frame.grid(row=11, column=0, columnspan=2, padx=30, pady=5, sticky="nws")

        camera_custom_setup = ttk.Frame(Frame_1)
        camera_custom_setup.grid(row=12, column=0, columnspan=2, padx=(30, 30), pady=5, sticky="w") 

        single_img = tk.Button(camera_custom_setup, text="Only Image", command=lambda: self.detect_single_img(camera_frame))
        single_img.grid(row=0, column=0, padx=(0, 10), pady=5, sticky="w", ipadx=5, ipady=2)
        single_img.config(state="disabled")
        self.lockable_widgets.append(single_img)

        multi_img = tk.Button(camera_custom_setup, text="Multi Image", command=lambda: self.detect_multi_img(camera_frame))
        multi_img.grid(row=0, column=1, padx=(0, 10), pady=5, sticky="w", ipadx=5, ipady=2)
        multi_img.config(state="disabled")
        self.lockable_widgets.append(multi_img)

        previos_img = tk.Button(camera_custom_setup, text="Prev...", command=lambda: self.detect_previos_img(camera_frame))
        previos_img.grid(row=0, column=2, padx=(0, 10), pady=5, sticky="w", ipadx=5, ipady=2)
        previos_img.config(state="disabled")
        self.lockable_widgets.append(previos_img)
        
        next_img = tk.Button(camera_custom_setup, text="Next...", command=lambda: self.detect_next_img(camera_frame))
        next_img.grid(row=0, column=3, padx=(0, 10), pady=5, sticky="w", ipadx=5, ipady=2)
        next_img.config(state="disabled")
        self.lockable_widgets.append(next_img)

        auto_detect = tk.Button(camera_custom_setup, text="Auto Detect", command=lambda: self.detect_auto(camera_frame))
        auto_detect.grid(row=0, column=4, padx=(0, 10), pady=5, sticky="w", ipadx=7, ipady=2)
        auto_detect.config(state="disabled")
        self.lockable_widgets.append(auto_detect)

        self.make_cls_var = tk.BooleanVar()
        make_cls = tk.Checkbutton(camera_custom_setup,text='Make class',variable=self.make_cls_var, onvalue=True, offvalue=False,anchor='w')
        make_cls.grid(row=0, column=5, padx=(0, 10), pady=5, sticky="w", ipadx=2, ipady=2)
        make_cls.config(state="disabled")
        self.lockable_widgets.append(make_cls)
        make_cls.var = self.make_cls_var

        logging_frame = ttk.Frame(Frame_1)
        logging_frame.grid(row=13, column=0, columnspan=2, padx=(30, 30), pady=10, sticky="w") 

        logging = tk.Button(logging_frame, text="Logging Image", command=lambda: self.logging(folder_ok,folder_ng,logging_ok_checkbox_var,logging_ng_checkbox_var,camera_frame,percent_entry,logging_frame))
        logging.grid(row=0, column=0, padx=(0,10), pady=5, sticky="w", ipadx=7, ipady=2)
        logging.config(state="disabled")
        self.lockable_widgets.append(logging)

        default_text_var = tk.StringVar()
        default_text_var.set("0%")
        percent_entry = ttk.Entry(logging_frame,textvariable=default_text_var,width=5)
        percent_entry.grid(row=0, column=1, padx=(0, 10), pady=3, sticky="w", ipadx=5, ipady=2)

        logging_ok_checkbox_var = tk.BooleanVar()
        logging_ok_checkbox = tk.Checkbutton(logging_frame,text='OK', variable=logging_ok_checkbox_var, onvalue=True, offvalue=False)
        logging_ok_checkbox.grid(row=1, column=2, padx=(0,10), pady=5, sticky="w", ipadx=7, ipady=2)
        logging_ok_checkbox.var = logging_ok_checkbox_var
        self.lock_params.append(logging_ok_checkbox)
        self.lockable_widgets.append(logging_ok_checkbox)

        logging_ng_checkbox_var = tk.BooleanVar()
        logging_ng_checkbox = tk.Checkbutton(logging_frame,text='NG', variable=logging_ng_checkbox_var, onvalue=True, offvalue=False)
        logging_ng_checkbox.grid(row=2, column=2, padx=(0,10), pady=5, sticky="w", ipadx=7, ipady=2)
        logging_ng_checkbox.var = logging_ng_checkbox_var
        self.lock_params.append(logging_ng_checkbox)
        self.lockable_widgets.append(logging_ng_checkbox)

        folder_ok = ttk.Entry(logging_frame, width=45)
        folder_ok.grid(row=1, column=0, padx=(0, 10), pady=3, sticky="w", ipadx=15, ipady=2)

        folder_ng = ttk.Entry(logging_frame ,width=45)
        folder_ng.grid(row=2, column=0, padx=(0, 10), pady=3, sticky="w", ipadx=15, ipady=2)

        folder_ok_button = tk.Button(logging_frame, text="Folder OK", command=lambda: self.pick_folder_ok(folder_ok))
        folder_ok_button.grid(row=1, column=1, padx=(0, 8), pady=3, sticky="w", ipadx=5, ipady=2)
        folder_ok_button.config(state="disabled")
        self.lockable_widgets.append(folder_ok_button)

        folder_ng_button = tk.Button(logging_frame, text="Folder NG", command=lambda: self.pick_folder_ng(folder_ng))
        folder_ng_button.grid(row=2, column=1, padx=(0, 8), pady=3, sticky="w", ipadx=5, ipady=2)
        folder_ng_button.config(state="disabled")
        self.lockable_widgets.append(folder_ng_button)

        move = tk.Button(logging_frame, text="action", command=lambda: self.mohica())
        move.grid(row=3, column=0, padx=(0, 8), pady=3, sticky="w", ipadx=5, ipady=2)

    def option_layout_parameters(self,Frame_2,model1):
        
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

        for i1 in range(len(model1.names)):
            row_widgets = []

            model_name_label = tk.Label(Frame_2, text=f'{model1.names[i1]}', fg='black', font=('Segoe UI', 12), width=15, anchor='w')
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
            ok_checkbox = tk.Checkbutton(Frame_2, variable=ok_checkbox_var, onvalue=True, offvalue=False, command=lambda rw=self.row_widgets:ok_selected(rw), anchor='w')
            ok_checkbox.grid()
            ok_checkbox.var = ok_checkbox_var
            row_widgets.append(ok_checkbox)
            self.ok_vars.append(ok_checkbox_var)
            self.lock_params.append(ok_checkbox)
            self.lockable_widgets.append(ok_checkbox)

            ng_checkbox_var = tk.BooleanVar()
            ng_checkbox = tk.Checkbutton(Frame_2, variable=ng_checkbox_var, onvalue=True, offvalue=False, command=lambda rw=self.row_widgets:ng_selected(rw), anchor='w')
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


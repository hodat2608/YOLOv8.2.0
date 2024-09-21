import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent.parent
ultralytics_main_dir = current_dir
sys.path.append(str(ultralytics_main_dir))
import tkinter as tk
from tkinter import ttk
import glob
# import stapipy as st
import shutil
import os,torch
from tkinter import messagebox,filedialog
import random,tqdm
from subprocess import Popen, PIPE, STDOUT
import threading
from base.config import *
import numpy as np
import cv2
from base.ultils import *

class Training_Data(Base):
    def __init__(self, *args, **kwargs):
        super(Training_Data, self).__init__(*args, **kwargs)
        super().__init__()
        torch.cuda.set_device(0)
        self.device_recognize = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_dir = os.getcwd()
        self.models_train = os.path.join(os.getcwd(),'base','setup.py')
        self.models_path= os.path.join(os.getcwd(),'ultralytics','cfg','models','v8')
        self.source_FOLDER_entry=None
        self.source_FOLDER_entry_btn=None
        self.source_CLASS_entry=None
        self.source_CLASS_entry_button=None
        self.size_model=None
        self.epochs_model=None
        self.batch_model=None
        self.device_model=None
        self.source_save_result_entry=None
        self.excute_button=None
        self.myclasses = []

    def format_params_xywhr2xyxyxyxy(self, des_path, progress_label):
        return super().format_params_xywhr2xyxyxyxy(des_path, progress_label)

    def layout(self,settings_notebook,window):
       
        canvas1 = tk.Canvas(settings_notebook)
        scrollbar = ttk.Scrollbar(settings_notebook, orient="vertical", command=canvas1.yview)
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

        settings_notebook.grid_columnconfigure(0, weight=1)
        settings_notebook.grid_rowconfigure(0, weight=1)

        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()

        frame_width = screen_width // 2
        frame_height = screen_height // 2


        Frame_1 = ttk.LabelFrame(settings_notebook, text="Configuration", width=frame_width, height=frame_height)
        Frame_2 = ttk.LabelFrame(settings_notebook, text="Console", width=frame_width, height=frame_height)

        Frame_1.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")  
        Frame_2.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
           
        scrollable_frame.grid_columnconfigure(0, weight=1)
        scrollable_frame.grid_columnconfigure(1, weight=1)
        scrollable_frame.grid_rowconfigure(0, weight=1)
        scrollable_frame.grid_rowconfigure(1, weight=1)

        ###############
        scrollbar = tk.Scrollbar(Frame_2)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.console_widget = tk.Text(Frame_2, height=50, width=150, bg='white', fg='black', insertbackground='white', yscrollcommand=scrollbar.set)
        self.console_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar.config(command=self.console_widget.yview)
        ###############

        canvas = tk.Canvas(Frame_1)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_frame_1 = tk.Scrollbar(Frame_1, orient="vertical", command=canvas.yview)
        scrollbar_frame_1.pack(side=tk.RIGHT, fill=tk.Y)

        canvas.config(yscrollcommand=scrollbar_frame_1.set)
        inner_frame_1 = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner_frame_1, anchor="nw")
        inner_frame_1.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))) 

        ###
        datasets_format = ttk.Frame(inner_frame_1)
        datasets_format.grid(row=1, column=0, columnspan=2, padx=(15, 30), pady=10, sticky="w") 

        ttk.Label(datasets_format, text='Dataset Formats:', font=('ubuntu', 12), width=15 ).grid(column=0, row=1, padx=10, pady=5, sticky="w")

        option_datasets_format = ['BASE Dataset Format','OBB Dataset Format']
        self.datasets_format_model = ttk.Combobox(datasets_format, values=option_datasets_format, width=7)
        self.datasets_format_model.grid(row=1, column=2, columnspan=2,  padx=(0, 10), pady=5, sticky="w", ipadx=40, ipady=2)

        ###
        source_FOLDER = ttk.Frame(inner_frame_1)
        source_FOLDER.grid(row=2, column=0, columnspan=2, padx=(15, 30), pady=10, sticky="w") 

        ttk.Label(source_FOLDER, text='Source folder:', font=('ubuntu', 12), width=15 ).grid(column=0, row=1, padx=10, pady=5, sticky="w")

        self.source_FOLDER_entry = ttk.Entry(source_FOLDER, width=45)
        self.source_FOLDER_entry.grid(row=1, column=1, padx=(0, 10), pady=3, sticky="w", ipadx=5, ipady=2)

        self.source_FOLDER_entry_btn = tk.Button(source_FOLDER, text="Browse...", command=lambda:self.browse_folder0())
        self.source_FOLDER_entry_btn.grid(row=1, column=2, padx=(0, 10), pady=3, sticky="w", ipadx=5, ipady=1)

        #####

        source_CLASS = ttk.Frame(inner_frame_1)
        source_CLASS.grid(row=3, column=0, columnspan=2, padx=(15, 30), pady=10, sticky="w") 

        ttk.Label(source_CLASS, text='Source class:', font=('ubuntu', 12),  width=15 ).grid(column=0, row=1, padx=10, pady=5, sticky="w")

        self.source_CLASS_entry = ttk.Entry(source_CLASS, width=45)
        self.source_CLASS_entry.grid(row=1, column=1, padx=(0, 10), pady=3, sticky="w", ipadx=5, ipady=2)

        self.source_CLASS_entry_button = tk.Button(source_CLASS, text="Browse...",command=lambda:self.browse_folder1())
        self.source_CLASS_entry_button.grid(row=1, column=2, padx=(0, 10), pady=3, sticky="w", ipadx=5, ipady=1)
                
        #####

        imgsz = ttk.Frame(inner_frame_1)
        imgsz.grid(row=4, column=0, columnspan=2, padx=(15, 30), pady=10, sticky="w") 

        ttk.Label(imgsz, text='Image size:', font=('ubuntu', 12), width=15).grid(row=1, column=1, padx=10, pady=5, sticky="w")
        
        options = [480, 608, 832]
        self.size_model = ttk.Combobox(imgsz, values=options, width=7)
        self.size_model.grid(row=1, column=2, columnspan=2,  padx=(0, 10), pady=5, sticky="w", ipadx=5, ipady=2)
        self.size_model.set(608)

        #####

        epochs = ttk.Frame(inner_frame_1)
        epochs.grid(row=5, column=0, columnspan=2, padx=(15, 30), pady=10, sticky="w") 

        ttk.Label(epochs, text='Epochs:', font=('ubuntu', 12), width=15).grid(row=1, column=1, padx=10, pady=5, sticky="w")
        
        optionsepochs = [100, 200, 300]
        self.epochs_model = ttk.Combobox(epochs, values=optionsepochs, width=7)
        self.epochs_model.grid(row=1, column=2, columnspan=2,  padx=(0, 10), pady=5, sticky="w", ipadx=5, ipady=2)
        self.epochs_model.set(300)

        #####

        time_frame = ttk.Frame(inner_frame_1)
        time_frame.grid(row=6, column=0, columnspan=2, padx=(15, 30), pady=10, sticky="w") 

        ttk.Label(time_frame, text='Times: ', font=('ubuntu', 12),  width=15 ).grid(column=0, row=1, padx=10, pady=5, sticky="w")

        self.time_frame_entry = ttk.Entry(time_frame, width=45)
        self.time_frame_entry.grid(row=1, column=1, padx=(0, 10), pady=3, sticky="w", ipadx=5, ipady=2)


        #####

        patience = ttk.Frame(inner_frame_1)
        patience.grid(row=7, column=0, columnspan=2, padx=(15, 30), pady=10, sticky="w") 

        ttk.Label(patience, text='Patience:', font=('ubuntu', 12), width=15).grid(row=1, column=1, padx=10, pady=5, sticky="w")
        
        optionsepochs = [100, 200, 300]
        self.patience_model = ttk.Combobox(patience, values=optionsepochs, width=7)
        self.patience_model.grid(row=1, column=2, columnspan=2,  padx=(0, 10), pady=5, sticky="w", ipadx=5, ipady=2)
        self.patience_model.set(300)  

        #####

        batch = ttk.Frame(inner_frame_1)
        batch.grid(row=8, column=0, columnspan=2, padx=(15, 30), pady=10, sticky="w") 

        ttk.Label(batch, text='Batch:', font=('ubuntu', 12), width=15).grid(row=1, column=1, padx=10, pady=5, sticky="w")
        
        optionsbatch = [2, 4, 8, 16, 24, 32]
        self.batch_model = ttk.Combobox(batch, values=optionsbatch, width=7)
        self.batch_model.grid(row=1, column=2, columnspan=2,  padx=(0, 10), pady=5, sticky="w", ipadx=5, ipady=2)
        self.batch_model.set(32)


        #####

        device = ttk.Frame(inner_frame_1)
        device.grid(row=9, column=0, columnspan=2, padx=(15, 30), pady=10, sticky="w") 

        ttk.Label(device, text='Device:', font=('ubuntu', 12), width=15).grid(row=1, column=1, padx=10, pady=5, sticky="w")
        
        optionsdevice = ['cpu','gpu','mps','Auto']
        self.device_model = ttk.Combobox(device, values=optionsdevice, width=7)
        self.device_model.grid(row=1, column=2, columnspan=2,  padx=(0, 10), pady=5, sticky="w", ipadx=5, ipady=2)
        self.device_model.set('Auto')

        #####

        source_save_result = ttk.Frame(inner_frame_1)
        source_save_result.grid(row=10, column=0, columnspan=2, padx=(15, 30), pady=10, sticky="w") 

        ttk.Label(source_save_result, text='Save Results:', font=('ubuntu', 12),  width=15 ).grid(column=0, row=1, padx=10, pady=5, sticky="w")

        self.source_save_result_entry = ttk.Entry(source_save_result, width=45)
        self.source_save_result_entry.grid(row=1, column=1, padx=(0, 10), pady=3, sticky="w", ipadx=5, ipady=2)

        self.source_save_result_entry_button = tk.Button(source_save_result, text="Browse...", command=lambda:self.browse_folder2())
        self.source_save_result_entry_button.grid(row=1, column=2, padx=(0, 10), pady=3, sticky="w", ipadx=5, ipady=1)

        #####

        name_results = ttk.Frame(inner_frame_1)
        name_results.grid(row=11, column=0, columnspan=2, padx=(15, 30), pady=10, sticky="w") 

        ttk.Label(name_results, text='Name Results:', font=('ubuntu', 12),  width=15 ).grid(column=0, row=1, padx=10, pady=5, sticky="w")

        self.name_results_entry = ttk.Entry(name_results, width=45)
        self.name_results_entry.grid(row=1, column=1, padx=(0, 10), pady=3, sticky="w", ipadx=5, ipady=2)


        # Save
        save_frame = ttk.Frame(inner_frame_1)
        save_frame.grid(row=12, column=0, columnspan=2, padx=(15, 30), pady=10, sticky="w")

        ttk.Label(save_frame, text='Save:', font=('ubuntu', 12), width=15).grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.save = tk.BooleanVar(value=True)
        ttk.Checkbutton(save_frame, variable=self.save).grid(row=1, column=2, sticky="w")


        # cache
        cache_frame = ttk.Frame(inner_frame_1)
        cache_frame.grid(row=13, column=0, columnspan=2, padx=(15, 30), pady=10, sticky="w")

        ttk.Label(cache_frame, text='Cache:', font=('ubuntu', 12), width=15).grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.cache = tk.BooleanVar(value=True)
        ttk.Checkbutton(cache_frame, variable=self.cache).grid(row=1, column=2, sticky="w")

        # Workers
        workers_frame = ttk.Frame(inner_frame_1)
        workers_frame.grid(row=14, column=0, columnspan=2, padx=(15, 30), pady=10, sticky="w")

        ttk.Label(workers_frame, text='Workers:', font=('ubuntu', 12), width=15).grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.workers = tk.IntVar(value=8)
        ttk.Spinbox(workers_frame, from_=0, to=32, textvariable=self.workers, width=7).grid(row=1, column=2, columnspan=2, padx=(0, 10), pady=5, sticky="w", ipadx=5, ipady=2)

        # Optimizer
        optimizer_frame = ttk.Frame(inner_frame_1)
        optimizer_frame.grid(row=15, column=0, columnspan=2, padx=(15, 30), pady=10, sticky="w")

        ttk.Label(optimizer_frame, text='Optimizer:', font=('ubuntu', 12), width=15).grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.optimizer_model = ttk.Combobox(optimizer_frame, values=["auto", "adam", "sgd"], width=7)
        self.optimizer_model.grid(row=1, column=2, columnspan=2, padx=(0, 10), pady=5, sticky="w", ipadx=5, ipady=2)
        self.optimizer_model.set("auto")

        # Pretrained
        pretrained_frame = ttk.Frame(inner_frame_1)
        pretrained_frame.grid(row=16, column=0, columnspan=2, padx=(15, 30), pady=10, sticky="w")

        ttk.Label(pretrained_frame, text='Pretrained:', font=('ubuntu', 12), width=15).grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.pretrained = tk.BooleanVar(value=True)
        ttk.Checkbutton(pretrained_frame, variable=self.pretrained).grid(row=1, column=2, sticky="w")


        # cache
        verbose_frame = ttk.Frame(inner_frame_1)
        verbose_frame.grid(row=17, column=0, columnspan=2, padx=(15, 30), pady=10, sticky="w")

        ttk.Label(verbose_frame, text='verbose:', font=('ubuntu', 12), width=15).grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.verbose = tk.BooleanVar(value=True)
        ttk.Checkbutton(verbose_frame, variable=self.verbose).grid(row=1, column=2, sticky="w")

        # cache
        deterministic_frame = ttk.Frame(inner_frame_1)
        deterministic_frame.grid(row=18, column=0, columnspan=2, padx=(15, 30), pady=10, sticky="w")

        ttk.Label(deterministic_frame, text='deterministic:', font=('ubuntu', 12), width=15).grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.deterministic = tk.BooleanVar(value=True)
        ttk.Checkbutton(deterministic_frame, variable=self.deterministic).grid(row=1, column=2, sticky="w")

        # cache
        single_cls_frame = ttk.Frame(inner_frame_1)
        single_cls_frame.grid(row=19, column=0, columnspan=2, padx=(15, 30), pady=10, sticky="w")

        ttk.Label(single_cls_frame, text='single_cls:', font=('ubuntu', 12), width=15).grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.single_cls = tk.BooleanVar(value=True)
        ttk.Checkbutton(single_cls_frame, variable=self.single_cls).grid(row=1, column=2, sticky="w")

         # cache
        rect_frame = ttk.Frame(inner_frame_1)
        rect_frame.grid(row=20, column=0, columnspan=2, padx=(15, 30), pady=10, sticky="w")

        ttk.Label(rect_frame, text='rect:', font=('ubuntu', 12), width=15).grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.rect = tk.BooleanVar(value=True)
        ttk.Checkbutton(rect_frame, variable=self.rect).grid(row=1, column=2, sticky="w")

        #####

        close_mosaic = ttk.Frame(inner_frame_1)
        close_mosaic.grid(row=21, column=0, columnspan=2, padx=(15, 30), pady=10, sticky="w") 

        ttk.Label(close_mosaic, text='close_mosaic:', font=('ubuntu', 12), width=15).grid(row=1, column=1, padx=10, pady=5, sticky="w")
        
        optionsbatch = [10, 20, 30, 40, 50, 100]
        self.close_mosaic_model = ttk.Combobox(close_mosaic, values=optionsbatch, width=7)
        self.close_mosaic_model.grid(row=1, column=2, columnspan=2,  padx=(0, 10), pady=5, sticky="w", ipadx=5, ipady=2)
        self.close_mosaic_model.set(10)

        ###
        resume_frame = ttk.Frame(inner_frame_1)
        resume_frame.grid(row=22, column=0, columnspan=2, padx=(15, 30), pady=10, sticky="w")

        ttk.Label(resume_frame, text='resume:', font=('ubuntu', 12), width=15).grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.resume = tk.BooleanVar(value=True)
        ttk.Checkbutton(resume_frame, variable=self.resume).grid(row=1, column=2, sticky="w")

        ###
        amp_frame = ttk.Frame(inner_frame_1)
        amp_frame.grid(row=23, column=0, columnspan=2, padx=(15, 30), pady=10, sticky="w")

        ttk.Label(amp_frame, text='amp:', font=('ubuntu', 12), width=15).grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.amp = tk.BooleanVar(value=True)
        ttk.Checkbutton(amp_frame, variable=self.amp).grid(row=1, column=2, sticky="w")

        ###
        fraction_frame = ttk.Frame(inner_frame_1)
        fraction_frame.grid(row=24, column=0, columnspan=2, padx=(15, 30), pady=10, sticky="w")

        ttk.Label(fraction_frame, text='fraction:', font=('ubuntu', 12), width=15).grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.fraction = tk.BooleanVar(value=True)
        ttk.Checkbutton(fraction_frame, variable=self.fraction).grid(row=1, column=2, sticky="w")

        ###
        profile_frame = ttk.Frame(inner_frame_1)
        profile_frame.grid(row=25, column=0, columnspan=2, padx=(15, 30), pady=10, sticky="w")

        ttk.Label(profile_frame, text='profile:', font=('ubuntu', 12), width=15).grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.profile = tk.BooleanVar(value=True)
        ttk.Checkbutton(profile_frame, variable=self.profile).grid(row=1, column=2, sticky="w")

        ####
        
        excute = ttk.Frame(inner_frame_1)
        excute.grid(row=26, column=0, columnspan=2, padx=(15, 30), pady=10, sticky="w") 

        self.excute_button = tk.Button(excute, text="Excute", command=lambda: self.Excute(progress_label))
        self.excute_button.grid(row=1, column=2, padx=(0, 10), pady=3, sticky="w", ipadx=15, ipady=1)

        progress_label = ttk.Label(excute, text="", font=('Segoe UI', 12))
        progress_label.grid(row=1, column=3, columnspan=2, padx=10, pady=5, sticky="nws")


    def browse_folder2(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.source_save_result_entry.delete(0, tk.END)
            self.source_save_result_entry.insert(0, folder_selected)

    def browse_folder1(self):
        file_selected = filedialog.askopenfilename(
            title="Select classes.txt file",
            filetypes=(("Text files", "classes.txt"),) 
        )
        if file_selected:
            self.source_CLASS_entry.delete(0, tk.END)
            self.source_CLASS_entry.insert(0, file_selected)

    def browse_folder0(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.source_FOLDER_entry.delete(0, tk.END)
            self.source_FOLDER_entry.insert(0, folder_selected)

    def run(self,progress_label):
        if self.source_FOLDER_entry.get() == None or self.source_FOLDER_entry.get() == '' or self.source_CLASS_entry.get() == None or self.source_CLASS_entry.get() == '':
            messagebox.showerror("Error", f"Please choose source folder datasets")
        else:    
            des_path = os.path.join(self.current_dir, 'datasets')
            src_path = self.source_FOLDER_entry.get()
            for root, dirs, file in os.walk(des_path, topdown=False):
                for dir_name, file_name in zip(dirs,file):
                    dir_path = os.path.join(root, dir_name)
                    file_path = os.path.join(root, file_name)
                    shutil.rmtree(dir_path)
                    os.remove(file_path)
            try:
                folders = ['train/images', 'train/labels', 'valid/images', 'valid/labels']
                for folder in folders:
                    os.makedirs(os.path.join(des_path, folder), exist_ok=True)
            except Exception as e:
                messagebox.showwarning("Warning", f'Can not create *train and ** valid path. Error: {e}')

            jpg = glob.glob(src_path + '/*.jpg')
            total_files = len(jpg)
            list_image_train = random.sample(jpg, int(len(jpg) * 0.85))

            for index, file_path in enumerate(jpg):
                file_name = os.path.basename(file_path)
                label_path = file_path[:-3] + 'txt'
                if file_path in list_image_train:
                    dest_image_folder = os.path.join(des_path, 'train', 'images')
                    dest_label_folder = os.path.join(des_path, 'train', 'labels')
                else:
                    dest_image_folder = os.path.join(des_path, 'valid', 'images')
                    dest_label_folder = os.path.join(des_path, 'valid', 'labels')
                shutil.copyfile(file_path, os.path.join(dest_image_folder, file_name))
                shutil.copyfile(label_path, os.path.join(dest_label_folder, file_name[:-3] + 'txt'))
                progress_retail = (index + 1) / total_files * 100
                progress_label.config(text=f"Split datasets in progress: {progress_retail:.2f}%")
                progress_label.update_idletasks()

            if os.path.exists(src_path + '/classes.txt'):
                shutil.copyfile(src_path + '/classes.txt', des_path + '/classes.txt')

            with open(self.source_CLASS_entry.get(),'r') as line:
                cls = line.read().split('\n')
                for text in cls:
                    self.myclasses.append(text)
            
            self.myclasses = [cls for cls in self.myclasses if cls]

            with open(os.path.join(self.models_path,'datasets.yaml'), "w", encoding='utf-8') as f:
                f.write('train: ' + os.path.join(os.getcwd() , 'datasets/train/images'))
                f.write('\n')
                f.write('val: ' + os.path.join(os.getcwd(), 'datasets/valid/images'))
                f.write('\n')
                f.write('nc: '  + str(len(self.myclasses)))     
                f.write('\n')
                f.write('names: '  + str(self.myclasses))      
            
            with open(os.path.join(self.models_path,'yolov8.yaml'), "w", encoding='utf-8') as f:
                f.write('nc: ' +  str(len(self.myclasses)) + '\n' + YOLOV8_YAML)

            if  self.device_model.get() == "Auto" :
                device_model = self.device_recognize
            else :
                device_model = self.device_model.get()
                
            callback = (
                f'python {self.models_train} --config "{os.path.join(self.models_path,"yolov8.yaml")}" '
                f'--data "{os.path.join(self.models_path,"datasets.yaml")}" --epochs {str(self.epochs_model.get())} '
                f'--imgsz {str(self.size_model.get())} --batch {str(self.batch_model.get())} '
                f'--device {str(device_model)} --project "{None if self.source_save_result_entry.get()==''else self.source_save_result_entry.get()}" '
                f'--name {str(self.name_results_entry.get())} '
                f'--workers {str(self.workers.get())} '
                )

            self.execute_command(callback)

    def race(self,progress_label):
        if self.source_FOLDER_entry.get() == None or self.source_FOLDER_entry.get() == '' or self.source_CLASS_entry.get() == None or self.source_CLASS_entry.get() == '':
            messagebox.showerror("Error", f"Please choose source folder datasets")
        else:    
            src_path = self.source_FOLDER_entry.get()
            des_path = os.path.join(self.current_dir, 'datasets')

            if not os.path.exists(des_path):
                os.makedirs(des_path)

            for root, dirs, files in os.walk(des_path):
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    shutil.rmtree(dir_path)
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    os.remove(file_path)

            for filename in os.listdir(src_path):
                if filename.endswith('.jpg') or filename.endswith('.txt'):
                    src_file = os.path.join(src_path, filename)
                    dst_file = os.path.join(des_path, filename)
                    shutil.copy2(src_file, dst_file)
            
            self.format_params_xywhr2xyxyxyxy(des_path,progress_label)

            try:
                folders = ['train/images', 'train/labels', 'valid/images', 'valid/labels']
                for folder in folders:
                    os.makedirs(os.path.join(des_path, folder), exist_ok=True)
            except Exception as e:
                messagebox.showwarning("Warning", f'Can not create *train and ** valid path. Error: {e}')

            jpg = glob.glob(des_path + '/*.jpg')
            total_files = len(jpg)
            list_image_train = random.sample(jpg, int(len(jpg) * 0.85))

            for index, file_path in enumerate(jpg):
                file_name = os.path.basename(file_path)
                label_path = file_path[:-3] + 'txt'
                if file_path in list_image_train:
                    dest_image_folder = os.path.join(des_path, 'train', 'images')
                    dest_label_folder = os.path.join(des_path, 'train', 'labels')
                else:
                    dest_image_folder = os.path.join(des_path, 'valid', 'images')
                    dest_label_folder = os.path.join(des_path, 'valid', 'labels')
                shutil.move(file_path, os.path.join(dest_image_folder, file_name))
                shutil.move(label_path, os.path.join(dest_label_folder, file_name[:-3] + 'txt'))
                progress_retail = (index + 1) / total_files * 100
                progress_label.config(text=f"Split datasets in progress: {progress_retail:.2f}%")
                progress_label.update_idletasks()

            if os.path.exists(src_path + '/classes.txt'):
                shutil.copyfile(src_path + '/classes.txt', des_path + '/classes.txt')

            with open(self.source_CLASS_entry.get(),'r') as line:
                cls = line.read().split('\n')
                for text in cls:
                    self.myclasses.append(text)

            self.myclasses = [cls for cls in self.myclasses if cls]
            with open(os.path.join(self.models_path,'datasets.yaml'), "w", encoding='utf-8') as f:
                f.write('train: ' + os.path.join(os.getcwd() , 'datasets/train/images'))
                f.write('\n')
                f.write('val: ' + os.path.join(os.getcwd(), 'datasets/valid/images'))
                f.write('\n')
                f.write('nc: '  + str(len(self.myclasses)))     
                f.write('\n')
                f.write('names: '  + str(self.myclasses))      
            
            with open(os.path.join(self.models_path,'yolov8.yaml'), "w", encoding='utf-8') as f:
                f.write('nc: ' +  str(len(self.myclasses)) + '\n' + YOLOV8_OBB_YAML)

            if  self.device_model.get() == "Auto" :
                device_model = self.device_recognize
            else :
                device_model = self.device_model.get()

            callback = (
                f'python {self.models_train} --config "{os.path.join(self.models_path,"yolov8.yaml")}" '
                f'--data "{os.path.join(self.models_path,"datasets.yaml")}" --epochs {str(self.epochs_model.get())} '
                f'--imgsz {str(self.size_model.get())} --batch {str(self.batch_model.get())} '
                f'--device {str(device_model)} --project "{None if self.source_save_result_entry.get()==''else self.source_save_result_entry.get()}" '
                f'--name {str(self.name_results_entry.get())} '
                f'--workers {str(self.workers.get())} '
                )

            self.execute_command(callback)

    def run_command(self,command):
        process = Popen(command, shell=True, stdout=PIPE, stderr=STDOUT, text=True, encoding='utf-8')
        
        for line in process.stdout:
            self.console_widget.insert(tk.END, line)
            self.console_widget.see(tk.END)  

        process.stdout.close()
        process.wait()

    def execute_command(self,callback):
        command = callback
        if command.startswith("pip install"):
            command = f"python -m {command}"
        threading.Thread(target=self.run_command, args=(command,)).start()

    def Excute(self, progress_label):
        dataset_format = self.datasets_format_model.get()
        if dataset_format == '' or dataset_format == 'None':
            messagebox.showwarning("Warning", 'Please choose Supported Dataset Formats')
        else:
            src_path = self.source_FOLDER_entry.get()
            if not os.path.isdir(src_path):
                messagebox.showwarning("Warning", "Invalid source folder.")
                return

            txt_files = [file for file in os.listdir(src_path) if file.endswith('.txt')]
            if not txt_files:
                messagebox.showwarning("Warning", "No .txt files found in the source folder.")
                return
            with open(os.path.join(src_path, txt_files[1]), 'r') as file:
                lines = file.readlines()
                for line in lines:
                    line = line.strip()
                    if dataset_format == 'BASE Dataset Format':
                        if "YOLO_OBB" in line:
                            messagebox.showwarning("Warning", 'Invalid Supported Dataset Formats')
                            return
                        else:
                            self.run(progress_label)
                            return                      
                    elif dataset_format == 'OBB Dataset Format':
                        if "YOLO_OBB" not in line:
                            messagebox.showwarning("Warning", 'Invalid Supported Dataset Formats')
                            return
                        else:
                            self.race(progress_label)
                            return


        








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

class Training_Data():
    def __init__(self, *args, **kwargs):
        torch.cuda.set_device(0)
        self.device_recognize = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.current_dir = os.getcwd()
        self.models_train = os.path.join(os.getcwd(),'combine','train_model.py')
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


        Frame_1 = ttk.LabelFrame(scrollable_frame, text="Configuration", width=frame_width, height=frame_height)
        Frame_2 = ttk.LabelFrame(scrollable_frame, text="Console Command Prompt", width=frame_width, height=frame_height)

        Frame_1.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")  
        Frame_2.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
           
        scrollable_frame.grid_columnconfigure(0, weight=1)
        scrollable_frame.grid_columnconfigure(1, weight=1)
        scrollable_frame.grid_rowconfigure(0, weight=1)
        scrollable_frame.grid_rowconfigure(1, weight=1)

        scrollbar = tk.Scrollbar(Frame_2)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.console_widget = tk.Text(Frame_2, height=50, width=150, bg='black', fg='white', insertbackground='white', yscrollcommand=scrollbar.set)
        self.console_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar.config(command=self.console_widget.yview)

        source_FOLDER = ttk.Frame(Frame_1)
        source_FOLDER.grid(row=1, column=0, columnspan=2, padx=(15, 30), pady=10, sticky="w") 

        ttk.Label(source_FOLDER, text='Source folder:', font=('ubuntu', 12), width=15 ).grid(column=0, row=1, padx=10, pady=5, sticky="w")

        self.source_FOLDER_entry = ttk.Entry(source_FOLDER, width=45)
        self.source_FOLDER_entry.grid(row=1, column=1, padx=(0, 10), pady=3, sticky="w", ipadx=5, ipady=2)
        self.source_FOLDER_entry.insert(0, "*train, **valid")

        self.source_FOLDER_entry_btn = tk.Button(source_FOLDER, text="Browse...", command=lambda:self.browse_folder0())
        self.source_FOLDER_entry_btn.grid(row=1, column=2, padx=(0, 10), pady=3, sticky="w", ipadx=5, ipady=1)

        #####

        source_CLASS = ttk.Frame(Frame_1)
        source_CLASS.grid(row=2, column=0, columnspan=2, padx=(15, 30), pady=10, sticky="w") 

        ttk.Label(source_CLASS, text='Source class:', font=('ubuntu', 12),  width=15 ).grid(column=0, row=1, padx=10, pady=5, sticky="w")

        self.source_CLASS_entry = ttk.Entry(source_CLASS, width=45)
        self.source_CLASS_entry.insert(0, "*.txt")
        self.source_CLASS_entry.grid(row=1, column=1, padx=(0, 10), pady=3, sticky="w", ipadx=5, ipady=2)

        self.source_CLASS_entry_button = tk.Button(source_CLASS, text="Browse...",command=lambda:self.browse_folder1())
        self.source_CLASS_entry_button.grid(row=1, column=2, padx=(0, 10), pady=3, sticky="w", ipadx=5, ipady=1)
                
        #####

        imgsz = ttk.Frame(Frame_1)
        imgsz.grid(row=3, column=0, columnspan=2, padx=(15, 30), pady=10, sticky="w") 

        ttk.Label(imgsz, text='Image size:', font=('ubuntu', 12), width=15).grid(row=1, column=1, padx=10, pady=5, sticky="w")
        
        options = [468, 608, 832]
        self.size_model = ttk.Combobox(imgsz, values=options, width=7)
        self.size_model.grid(row=1, column=2, columnspan=2,  padx=(0, 10), pady=5, sticky="w", ipadx=5, ipady=2)
        self.size_model.set(608)

        #####

        epochs = ttk.Frame(Frame_1)
        epochs.grid(row=4, column=0, columnspan=2, padx=(15, 30), pady=10, sticky="w") 

        ttk.Label(epochs, text='Epochs:', font=('ubuntu', 12), width=15).grid(row=1, column=1, padx=10, pady=5, sticky="w")
        
        optionsepochs = [100, 200, 300]
        self.epochs_model = ttk.Combobox(epochs, values=optionsepochs, width=7)
        self.epochs_model.grid(row=1, column=2, columnspan=2,  padx=(0, 10), pady=5, sticky="w", ipadx=5, ipady=2)
        self.epochs_model.set(300)

        #####

        batch = ttk.Frame(Frame_1)
        batch.grid(row=5, column=0, columnspan=2, padx=(15, 30), pady=10, sticky="w") 

        ttk.Label(batch, text='Batch:', font=('ubuntu', 12), width=15).grid(row=1, column=1, padx=10, pady=5, sticky="w")
        
        optionsbatch = [2, 4, 8, 16, 24, 32]
        self.batch_model = ttk.Combobox(batch, values=optionsbatch, width=7)
        self.batch_model.grid(row=1, column=2, columnspan=2,  padx=(0, 10), pady=5, sticky="w", ipadx=5, ipady=2)
        self.batch_model.set(32)


        #####

        device = ttk.Frame(Frame_1)
        device.grid(row=6, column=0, columnspan=2, padx=(15, 30), pady=10, sticky="w") 

        ttk.Label(device, text='Device:', font=('ubuntu', 12), width=15).grid(row=1, column=1, padx=10, pady=5, sticky="w")
        
        optionsdevice = ['cpu','gpu','mps','Auto']
        self.device_model = ttk.Combobox(device, values=optionsdevice, width=7)
        self.device_model.grid(row=1, column=2, columnspan=2,  padx=(0, 10), pady=5, sticky="w", ipadx=5, ipady=2)
        self.device_model.set('Auto')

        #####

        source_save_result = ttk.Frame(Frame_1)
        source_save_result.grid(row=7, column=0, columnspan=2, padx=(15, 30), pady=10, sticky="w") 

        ttk.Label(source_save_result, text='Save results:', font=('ubuntu', 12),  width=15 ).grid(column=0, row=1, padx=10, pady=5, sticky="w")

        self.source_save_result_entry = ttk.Entry(source_save_result, width=45)
        self.source_save_result_entry.insert(0, "Select folder to save the results")
        self.source_save_result_entry.grid(row=1, column=1, padx=(0, 10), pady=3, sticky="w", ipadx=5, ipady=2)

        self.source_save_result_entry_button = tk.Button(source_save_result, text="Browse...", command=lambda:self.browse_folder2())
        self.source_save_result_entry_button.grid(row=1, column=2, padx=(0, 10), pady=3, sticky="w", ipadx=5, ipady=1)

        ####
        excute = ttk.Frame(Frame_1)
        excute.grid(row=8, column=0, columnspan=2, padx=(15, 30), pady=10, sticky="w") 

        self.excute_button = tk.Button(excute, text="Excute", command=lambda: self.Execute_Command_Prompt())
        self.excute_button.grid(row=1, column=2, padx=(0, 10), pady=3, sticky="w", ipadx=15, ipady=1)

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

    def Execute_Command_Prompt(self):
        if self.source_FOLDER_entry == None or self.source_CLASS_entry == None:
            messagebox.showerror("Error", f"Please choose source folder datasets")
        else:    
            des_path = os.path.join(self.current_dir, 'datasets')
            src_path = self.source_FOLDER_entry.get()
            try:
                os.makedirs(os.path.join(des_path,'train'),exist_ok=True)
                os.makedirs(os.path.join(des_path,'valid'),exist_ok=True)
                os.makedirs(os.path.join(des_path,'train','images'),exist_ok=True)
                os.makedirs(os.path.join(des_path,'valid','images'),exist_ok=True)
                os.makedirs(os.path.join(des_path,'train','labels'),exist_ok=True)
                os.makedirs(os.path.join(des_path,'valid','labels'),exist_ok=True)
            except Exception as e:
                messagebox.showwarning("Warning", f'Can not create *train and ** valid path. Error: {e}')

            jpg = glob.glob(src_path + '/*.jpg')
            list_image_train = random.sample(jpg, int(len(jpg) * 0.85))

            for i in tqdm.tqdm(jpg, desc="Proceed with data split...", unit="file"):
                tenf = os.path.basename(i)
                if i in list_image_train:
                    shutil.copyfile(i, des_path + '/train/images/' + tenf)
                    shutil.copyfile(i[:-3] + 'txt', des_path + '/train/labels/' + tenf[:-3] + 'txt')
                else:
                    shutil.copyfile(i, des_path + '/valid/images/' + tenf)
                    shutil.copyfile(i[:-3] + 'txt', des_path + '/valid/labels/' + tenf[:-3] + 'txt')
            if os.path.exists(src_path + '/classes.txt'):
                shutil.copyfile(src_path + '/classes.txt', des_path + '/classes.txt')

            with open(self.source_CLASS_entry.get(),'r') as line:
                cls = line.read().split('\n')
                for text in cls:
                    self.myclasses.append(text)
            
            for i in self.myclasses:
                if i == '':
                    continue
                self.myclasses.clear()
                self.myclasses.append(i)

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
                f'--device {str(device_model)} --project "{self.source_save_result_entry.get()}"'
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











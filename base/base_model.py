import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent.parent
ultralytics_main_dir = current_dir
sys.path.append(str(ultralytics_main_dir))
from ultralytics import YOLO
from tkinter import filedialog
from PIL import Image, ImageTk
import glob
import mysql.connector
from tkinter import messagebox,simpledialog
import threading
import numpy as np
import time
from yaml import load
import socket
from udp import UDPFinsConnection
from initialization import FinsPLCMemoryAreas
import tkinter as tk
import shutil
import os
soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
from collections import defaultdict
import cv2
from ultralytics.utils.files import increment_path
import supervision as sv
from collections import deque
from ultralytics.utils.plotting import Annotator, colors
import torch
def removefile():
    directory1 = 'C:/Users/CCSX009/Documents/yolov5/test_image/camera1/*.jpg'
    directory2 = 'C:/Users/CCSX009/Documents/yolov5/test_image/camera2/*.jpg'
    chk1 = glob.glob(directory1)
    for f1 in chk1:
        os.remove(f1)
        print('already delete folder 1')
    chk2 = glob.glob(directory2)
    for f2 in chk2:
        os.remove(f2)
        print('already delete folder 2')


class MySQL_Connection():

    def __init__(self,host,user,passwd,database):
        self.host = host
        self.user = user
        self.passwd = passwd
        self.database = database

    # def Connect_MySQLServer(self):
    #         db_connection = mysql.connector.connect(
    #         host= self.host,
    #         user=self.user, 
    #         passwd=self.passwd,
    #         database=self.database)     
    #         cursor = db_connection.cursor()
    #         return cursor,db_connection

    def check_connection(self):
        _,db_connection = self.Connect_MySQLServer()
        try:
            db_connection.ping(reconnect=True, attempts=3, delay=5)
            return True
        except mysql.connector.Error as err:
            messagebox.showinfo("Notification", f"Error connecting to the database: {str(err)}")
            return False

    def reconnect(self):
        _,db_connection = self.Connect_MySQLServer()
        try:
            db_connection.reconnect(attempts=3, delay=5)
            cursor = db_connection.cursor()  
            return True
        except mysql.connector.Error as err:
            messagebox.showinfo("Notification", f"Failed to reconnect to the database: {str(err)}")
            return False
        
    @staticmethod    
    def Connect_to_MySQLServer(host,user,passwd,database):
            db_connection = mysql.connector.connect(
            host= host,
            user=user, 
            passwd=passwd,
            database=database)                    
            cursor = db_connection.cursor()
            return cursor,db_connection
    
    def Connect_MySQLServer(self):
        try:
            db_connection = mysql.connector.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.passwd
            )
            if db_connection.is_connected():
                print("Successfully connected to the database")
                cursor = db_connection.cursor(dictionary=True)
                return cursor, db_connection
        except Exception as e:
            print(f"Error: {e}")
            return None, None
      

class PLC_Connection():

    def __init__(self,host,port):
        self.host = host
        self.port = port
      
    def connect_plc_keyence(self):
        try:
            soc.connect((self.host, self.port))
            return True
        except OSError:
            print("Can't connect to PLC")
            time.sleep(3)
            print("Reconnecting....")
            return False

    def run_plc_keyence(self):
        connected = False
        while connected == False:
            connected = self.connect_plc_keyence(self.host,self.port)
        print("connected") 

    def read_plc_keyence(self,data):
        a = 'RD '
        c = '\x0D'
        d = a+ data +c
        datasend = d.encode("UTF-8")
        soc.sendall(datasend)
        data = soc.recv(1024)
        datadeco = data.decode("UTF-8")
        data1 = int(datadeco)
        return data1

    def write_plc_keyence(self,register,data):
        a = 'WR '
        b = ' '
        c = '\x0D'
        d = a+register+b+str(data)+c
        datasend  = d.encode("UTF-8")
        soc.sendall(datasend)
        datares = soc.recv(1024)

    def connect_plc_omron(self):
        global fins_instance
        try:
            fins_instance = UDPFinsConnection()
            fins_instance.connect(self.host)
            fins_instance.dest_node_add=1
            fins_instance.srce_node_add=25
            return True
        except:
            print("Can't connect to PLC")
            for i in range(100000000):
                pass
            print("Reconnecting....")
            return False

    def run_plc_omron(self):
        connected = False
        while connected == False:
            connected = self.connect_plc_omron(self.host)
            print('connecting ....')
        print("connected plc") 

    def read_plc_omron(self,register):
        register = (register).to_bytes(2, byteorder='big') + b'\x00'
        read_var = fins_instance.memory_area_read(FinsPLCMemoryAreas().DATA_MEMORY_WORD,register)
        read_var = int.from_bytes(read_var[-2:], byteorder='big')  
        return read_var

    def write_plc_omron(self,register,data):
        register = (register).to_bytes(2, byteorder='big') + b'\x00'
        fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,register,b'\x00\x00',data)



class Base:

    def __init__(self):
        self.database = MySQL_Connection(None,None,None,None) 
        self.name_table = None
        self.item_code_cfg = None
        self.image_files = []
        self.current_image_index = 0
        self.state = 0
        self.password = " "
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
        self.make_cls_var = False
        self.permisson_btn = []
        self.model = None
        self.time_processing_output = None
        self.result_detection = None

    def connect_database(self):
        cursor, db_connection  = self.database.Connect_MySQLServer()
        check_connection = self.database.check_connection()
        reconnect = self.database.reconnect()
        return cursor,db_connection,check_connection,reconnect
    
    def check_connect_database(self):
        cursor, db_connection = self.database.Connect_MySQLServer()
        if cursor is not None and db_connection is not None:
            print("Database connection successful")
        else:
            print("Database connection failed")
        return cursor, db_connection
    
    def save_params_model(self):
        confirm_save_data = messagebox.askokcancel("Confirm", "Are you sure you want to save the data?")
        cursor,db_connection,check_connection,reconnect = self.connect_database()
        if confirm_save_data:
            try:
                weight = self.weights.get()
                confidence_all = int(self.scale_conf_all.get())
                size_model = int(self.size_model.get())
                item_code_value = str(self.item_code.get())
                cursor.execute(f"DELETE FROM {self.name_table} WHERE item_code = %s", (item_code_value,))
                for i1 in range(len(self.model_name_labels)):
                    label_name =  self.model_name_labels[i1].cget("text")
                    join_detect = self.join[i1].get()
                    OK_jont = self.ok_vars[i1].get()
                    NG_jont = self.ng_vars[i1].get()
                    num_labels = int(self.num_inputs[i1].get())
                    width_min = int(self.wn_inputs[i1].get())
                    width_max = int(self.wx_inputs[i1].get())
                    height_min = int(self.hn_inputs[i1].get())
                    height_max = int(self.hx_inputs[i1].get())
                    PLC_value = int(self.plc_inputs[i1].get())
                    cmpnt_conf = int(self.conf_scales[i1].get())
                    query_sql = f"""
                    INSERT INTO {self.name_table}
                    (item_code, weight, confidence_all, label_name,join_detect, OK, NG, num_labels, width_min, width_max, 
                    height_min, height_max, PLC_value, cmpnt_conf, size_detection)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    values = (item_code_value, weight, confidence_all, label_name, join_detect, OK_jont, NG_jont, num_labels, 
                            width_min, width_max, height_min, height_max, PLC_value, cmpnt_conf,size_model)
                    cursor.execute(query_sql,values)
                db_connection.commit()
                cursor.close()
                db_connection.close()
                messagebox.showinfo("Notification", "Saved parameters successfully!")
            except Exception as e:
                cursor.close()
                db_connection.close()
                messagebox.showerror("Error", f"Data saved failed! Error: {str(e)}")
        else:
            pass

    def processing_handle_image_customize(self, input_image, width, height):
        size_model_all = int(self.size_model.get())
        conf_all = int(self.scale_conf_all.get()) / 100
        results = self.model(input_image,imgsz=size_model_all,conf=conf_all)
        model_settings = [
            {
                'label_name':  self.model_name_labels[i1].cget("text"),
                'join_detect': self.join[i1].get(),
                'OK_jont': self.ok_vars[i1].get(),
                'NG_jont': self.ng_vars[i1].get(),
                'num_labels': int(self.num_inputs[i1].get()),
                'width_min': int(self.wn_inputs[i1].get()),
                'width_max': int(self.wx_inputs[i1].get()),
                'height_min': int(self.hn_inputs[i1].get()),
                'height_max': int(self.hx_inputs[i1].get()),
                'PLC_value': int(self.plc_inputs[i1].get()),
                'cmpnt_conf': int(self.conf_scales[i1].get()),
            }
            for i1 in range(len(self.model_name_labels))
        ]
        settings_dict = {setting['label_name']: setting for setting in model_settings}
        boxes_dict = results[0].boxes.cpu().numpy()
        xywh_list = boxes_dict.xywh.tolist()
        cls_list = boxes_dict.cls.tolist()
        conf_list = boxes_dict.conf.tolist()
        allowed_classes,list_remove,list_label_ng,ok_variable,results_detect= [],[],[],False,'ERROR'
        for xywh, cls, conf in zip(xywh_list, cls_list, conf_list):
            setting = settings_dict[results[0].names[int(cls)]]
            if setting:
                if setting['join_detect']:
                    if xywh[2] < setting['width_min'] or xywh[2] > setting['width_max'] \
                            or xywh[3] < setting['height_min'] or xywh[3] > setting['height_max'] \
                            or int(conf*100) < setting['cmpnt_conf']:
                        list_remove.append(int(cls))
                        continue
                    allowed_classes.append(results[0].names[int(cls)])
                else:
                    list_remove.append(int(cls))           
        for model_name,setting in settings_dict.items():
            if setting['join_detect'] and setting['OK_jont']:
                if allowed_classes.count(setting['label_name']) != setting['num_labels']:
                    results_detect,ok_variable = 'NG',True
                    list_label_ng.append(model_name)
            if setting['join_detect'] and setting['NG_jont']:
                if model_name in allowed_classes:
                    results_detect,ok_variable = 'NG',True
                    list_label_ng.append(setting['label_name'])
        if not ok_variable:
            results_detect = 'OK'

        show_img = np.squeeze(results[0].extract_npy(list_remove=list_remove))
        show_img = cv2.resize(show_img, (width, height), interpolation=cv2.INTER_AREA)
        output_image = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
        return output_image, results_detect, list_label_ng

    
    def load_data_model(self):
        cursor, db_connection,_,_ = self.connect_database()
        print(self.name_table)
        cursor.execute(f"SELECT * FROM {self.name_table} WHERE item_code = %s", (self.item_code_cfg,))
        records = cursor.fetchall()
        cursor.close()
        db_connection.close()
        if records:
            first_record = records[0]
            load_item_code = first_record['item_code']
            load_path_weight = first_record['weight']
            load_confidence_all_scale = first_record['confidence_all']
        return records,load_path_weight,load_item_code,load_confidence_all_scale

    def load_parameters_model(self,model1,load_path_weight,load_item_code,load_confidence_all_scale,records):
        self.weights.delete(0, tk.END)
        self.weights.insert(0, load_path_weight)
        self.item_code.delete(0, tk.END)
        self.item_code.insert(0, load_item_code)
        self.scale_conf_all.set(load_confidence_all_scale)
        try:
            for i1 in range(len(model1.names)):          
                for record in records:                
                    if record['label_name'] == model1.names[i1]:
                        self.join[i1].set(bool(record['join_detect']))
                        self.ok_vars[i1].set(bool(record['OK']))
                        self.ng_vars[i1].set(bool(record['NG']))
                        self.num_inputs[i1].delete(0, tk.END)
                        self.num_inputs[i1].insert(0, record['num_labels'])
                        self.wn_inputs[i1].delete(0, tk.END)
                        self.wn_inputs[i1].insert(0, record['width_min'])
                        self.wx_inputs[i1].delete(0, tk.END)
                        self.wx_inputs[i1].insert(0, record['width_max'])
                        self.hn_inputs[i1].delete(0, tk.END)
                        self.hn_inputs[i1].insert(0, record['height_min'])                
                        self.hx_inputs[i1].delete(0, tk.END)
                        self.hx_inputs[i1].insert(0, record['height_max'])
                        self.plc_inputs[i1].delete(0, tk.END)
                        self.plc_inputs[i1].insert(0, record['PLC_value'])
                        self.conf_scales[i1].set(record['cmpnt_conf'])
        except IndexError as e:
            messagebox.showerror("Error", f"Load parameters failed! Error: {str(e)}")

    def change_model(self,Frame_2):
        selected_file = filedialog.askopenfilename(title="Choose a file", filetypes=[("Model Files", "*.pt")])
        if selected_file:
            self.weights.delete(0,tk.END)
            self.weights.insert(0,selected_file)
            self.model = YOLO(selected_file)
            for widget in Frame_2.grid_slaves():
                widget.grid_forget()
            self.option_layout_parameters(Frame_2,self.model)
        else:
            messagebox.showinfo("Notification","Please select the correct training file!")
            pass

    def load_params_child(self):
        weight = self.weights.get()
        item_code_value = str(self.item_code.get())
        cursor, db_connection,_,_ = self.connect_database()
        cursor.execute(f"SELECT * FROM {self.name_table} WHERE item_code = %s", (item_code_value,))
        records = cursor.fetchall()
        model = YOLO(weight)
        cursor.close()
        db_connection.close()
        return records,model

    def load_parameters_from_weight(self, records):
        confirm_load_parameters = messagebox.askokcancel("Confirm", "Are you sure you want to load the parameters?")
        if confirm_load_parameters:
            records, model = self.load_params_child()
            try:
                for i1 in range(len(model.names)):          
                    for record in records:                
                        if record['label_name'] == model.names[i1]:
                            self.join[i1].set(bool(record['join_detect']))
                            self.ok_vars[i1].set(bool(record['OK']))
                            self.ng_vars[i1].set(bool(record['NG']))
                            self.num_inputs[i1].delete(0, tk.END)
                            self.num_inputs[i1].insert(0, record['num_labels'])
                            self.wn_inputs[i1].delete(0, tk.END)
                            self.wn_inputs[i1].insert(0, record['width_min'])
                            self.wx_inputs[i1].delete(0, tk.END)
                            self.wx_inputs[i1].insert(0, record['width_max'])
                            self.hn_inputs[i1].delete(0, tk.END)
                            self.hn_inputs[i1].insert(0, record['height_min'])                
                            self.hx_inputs[i1].delete(0, tk.END)
                            self.hx_inputs[i1].insert(0, record['height_max'])
                            self.plc_inputs[i1].delete(0, tk.END)
                            self.plc_inputs[i1].insert(0, record['PLC_value'])
                            self.conf_scales[i1].set(record['cmpnt_conf'])
            except IndexError as e:
                messagebox.showerror("Error", f"Load parameters failed! Error: {str(e)}")

    def handle_image(self,img1_orgin, width, height,camera_frame,cls):
        t1 = time.time()
        for widget in camera_frame.winfo_children():
            widget.destroy()
        image_result,results_detect,label_ng = self.processing_handle_image_local(img1_orgin, width, height,clss=cls)
        t2 = time.time() - t1
        time_processing = str(int(t2*1000)) + 'ms'
        img_pil = Image.fromarray(image_result)
        photo = ImageTk.PhotoImage(img_pil)
        canvas = tk.Canvas(camera_frame, width=width, height=height)
        canvas.grid(row=2, column=0, padx=10, pady=10, sticky='ew')
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo
        canvas.create_text(10, 10, anchor=tk.NW, text=f'Time: {time_processing}', fill='black', font=('Segoe UI', 20))
        canvas.create_text(10, 40, anchor=tk.NW, text=f'Result: {results_detect}', fill='green' if results_detect == 'OK' else 'red', font=('Segoe UI', 20))
        if not label_ng:
            canvas.create_text(10, 70, anchor=tk.NW, text=f'No Label', fill='green', font=('Segoe UI', 20))
        else:
            label_ng = ','.join(label_ng)
            canvas.create_text(10, 70, anchor=tk.NW, text=f'Label: {label_ng}', fill='red', font=('Segoe UI', 20))
        return results_detect  
    
    def detect_single_img(self, camera_frame):
        selected_file = filedialog.askopenfilename(title="Choose a file", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if selected_file:
            for widget in camera_frame.winfo_children():
                widget.destroy()
            width = 480
            height = 450
            cls = True if self.make_cls_var.get() else False
            self.handle_image(selected_file, width, height,camera_frame,cls)
        else: 
            pass
           
    def detect_multi_img(self,camera_frame):
        selected_folder = filedialog.askdirectory(title="Choose a folder")
        if selected_folder:
            self.image_files = [os.path.join(selected_folder, f) for f in os.listdir(selected_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            self.current_image_index = 0
            if self.image_files:
                for widget in camera_frame.winfo_children():
                    widget.destroy()
                self.show_image(self.current_image_index,camera_frame)
            else:
                messagebox.showinfo("No Images", "The selected folder contains no images.")
        else:
            pass

    def show_image(self,index,camera_frame):  
        width = 480
        height = 450
        image_path = self.image_files[index]
        cls = True if self.make_cls_var.get() else False
        self.handle_image(image_path, width, height,camera_frame,cls)

    def detect_next_img(self,camera_frame):
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.show_image(self.current_image_index,camera_frame)
        else:
            messagebox.showinfo("End of Images", "No more images in the folder.")

    def detect_previos_img(self,camera_frame):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_image(self.current_image_index,camera_frame)
        else:
            messagebox.showinfo("Start of Images", "This is the first image in the folder.")

    def detect_auto(self, camera_frame):
        selected_folder_original = filedialog.askdirectory(title="Choose a folder")
        if selected_folder_original: 
            selected_folder = glob.glob(selected_folder_original + '/*.jpg')
            if not selected_folder:
                pass
            self.image_index = 0
            self.selected_folder_detect_auto = selected_folder
            self.camera_frame = camera_frame
            self.image_path_mks_cls = []
            self.results_detect = None
            def process_next_image():
                if self.image_index < len(self.selected_folder_detect_auto):
                    self.image_path_mks_cls = self.selected_folder_detect_auto[self.image_index]
                    width = 480
                    height = 450
                    cls = True if self.make_cls_var.get() else False
                    self.results_detect = self.handle_image(self.image_path_mks_cls, width, height,camera_frame,cls)
                    self.image_index += 1
                    self.camera_frame.after(500, process_next_image)
                else:
                    print("Processing auto detect complete")
            process_next_image()
        else:
            pass

    
    def logging(self, folder_ok, folder_ng, logging_ok_checkbox_var, logging_ng_checkbox_var, camera_frame, percent_entry,logging_frame):
        selected_folder_original = filedialog.askdirectory(title="Choose a folder")
        selected_folder = glob.glob(os.path.join(selected_folder_original,'*.jpg'))
        width = 480
        height = 450
        self.image_index = 0
        self.selected_folder_logging = selected_folder
        self.logging_frame = logging_frame
        self.percent_entry = percent_entry
        total_images = len(self.selected_folder_logging)
        if not self.selected_folder_logging:
            return
        
        def update_progress(current_index, total_images):
            percent_value = int((current_index + 1) / total_images * 100)
            self.percent_entry.delete(0, tk.END)
            self.percent_entry.insert(0, f"{percent_value}%")

        def process_images():
            for img in self.selected_folder_logging:
                basename = os.path.basename(img)
                if self.image_index < total_images:
                    results_detect = self.handle_image(img, width, height, camera_frame)
                    if results_detect == 'OK':
                        if logging_ok_checkbox_var.get():
                            shutil.move(img, os.path.join(folder_ok.get(), basename))
                    else:
                        if logging_ng_checkbox_var.get():
                            shutil.move(img, os.path.join(folder_ng.get(), basename))
                    self.logging_frame.after(10, update_progress, self.image_index, total_images)
                    self.image_index += 1
                else:
                    messagebox.showinfo("End of Images", "No more images in the folder.")
                    break
            self.percent_entry.delete(0, tk.END)
            self.percent_entry.insert(0, "0%")
        threading.Thread(target=process_images).start()
        
             
    def toggle_state_layout_model(self): 
        if self.state == 1:
            password = simpledialog.askstring("Administrator", "Enter password:", show="*")
            if password == self.password:
                self.state = 0
                self.permisson_btn.config(text="Lock")
                self.toggle_widgets_state("normal")
            else:
                messagebox.showerror("Error", "Incorrect password!")
        else:
            self.state = 1
            self.permisson_btn.config(text="Unlock")
            self.toggle_widgets_state("disabled")

    def toggle_widgets_state(self,state):
        for widget in self.lockable_widgets:
            widget.config(state=state)

    def toggle_state_option_layout_parameters(self):
        for widget in self.lock_params:
            widget.config(state='disabled')

    def pick_folder_ok(self,folder_ok):
        file_path = filedialog.askdirectory(title="Choose a folder")
        if file_path:
            folder_ok.delete(0,tk.END)
            folder_ok.insert(0,file_path)
           
    def pick_folder_ng(self,folder_ng):
        file_path = filedialog.askdirectory(title="Choose a folder")
        if file_path:
            folder_ng.delete(0,tk.END)
            folder_ng.insert(0,file_path)


    def _make_cls(self,image_path_mks_cls,results,model_settings):  
        with open(image_path_mks_cls[:-3] + 'txt', "a") as file:
            for params in results.xywhn:
                params = params.tolist()
                for item in range(len(params)):
                    param = params[item]
                    param = [round(i,6) for i in param]
                    number_label = int(param[5])
                    conf_result = float(param[4])
                    width_result = float(param[2])*1200
                    height_result = float(param[3])*1600
                    for setting in model_settings:
                        if results.names[int(number_label)] == setting['label_name']:
                            if setting['join_detect']: 
                                if width_result < setting['width_min'] or width_result > setting['width_max'] \
                                        or height_result < setting['height_min'] or height_result > setting['height_max'] \
                                        or conf_result < setting['cmpnt_conf']: 
                                    formatted_values = ["{:.6f}".format(value) for value in param[:4]]
                                    output_line = "{} {}\n".format(str(number_label),' '.join(formatted_values))
                                    file.write(output_line)
        path = Path(image_path_mks_cls).parent
        path = os.path.join(path,'classes.txt')
        with open(path, "w") as file:
            for i1 in range(len(results.names)):
                file.write(str(results.names[i1])+'\n')
      
    def classify_imgs(self):
        pass    

    def processing_handle_image_local(self,input_image_original,width,height,clss=False):
        size_model_all = int(self.size_model.get())
        conf_all = int(self.scale_conf_all.get()) / 100
        results = self.model(input_image_original,imgsz=size_model_all,conf=conf_all)
        model_settings = [
            {
                'label_name':  self.model_name_labels[i1].cget("text"),
                'join_detect': self.join[i1].get(),
                'OK_jont': self.ok_vars[i1].get(),
                'NG_jont': self.ng_vars[i1].get(),
                'num_labels': int(self.num_inputs[i1].get()),
                'width_min': int(self.wn_inputs[i1].get()),
                'width_max': int(self.wx_inputs[i1].get()),
                'height_min': int(self.hn_inputs[i1].get()),
                'height_max': int(self.hx_inputs[i1].get()),
                'PLC_value': int(self.plc_inputs[i1].get()),
                'cmpnt_conf': int(self.conf_scales[i1].get()),
            }
            for i1 in range(len(self.model_name_labels))
        ]
        settings_dict = {setting['label_name']: setting for setting in model_settings}
        boxes_dict = results[0].boxes.cpu().numpy()
        xywh_list = boxes_dict.xywh.tolist()
        cls_list = boxes_dict.cls.tolist()
        conf_list = boxes_dict.conf.tolist()
        allowed_classes,list_remove,list_label_ng,ok_variable,results_detect= [],[],[],False,'ERROR'
        for xywh, cls, conf in zip(xywh_list, cls_list, conf_list):
            setting = settings_dict[results[0].names[int(cls)]]
            if setting:                                       
                if setting['join_detect']:
                    if xywh[2] < setting['width_min'] or xywh[2] > setting['width_max'] \
                            or xywh[3] < setting['height_min'] or xywh[3] > setting['height_max'] \
                            or int(conf*100) < setting['cmpnt_conf']:
                        list_remove.append(int(cls))
                        continue
                    allowed_classes.append(results[0].names[int(cls)])
                else:
                    list_remove.append(int(cls))           
        for model_name,setting in settings_dict.items():
            if setting['join_detect'] and setting['OK_jont']:
                if allowed_classes.count(setting['label_name']) != setting['num_labels']:
                    results_detect,ok_variable = 'NG',True
                    list_label_ng.append(model_name)
            if setting['join_detect'] and setting['NG_jont']:
                if model_name in allowed_classes:
                    results_detect,ok_variable = 'NG',True
                    list_label_ng.append(setting['label_name'])
        if not ok_variable:
            results_detect = 'OK'
        if clss:
            self._make_cls(input_image_original,results,model_settings)
        show_img = np.squeeze(results[0].extract_npy(list_remove=list_remove))
        show_img = cv2.resize(show_img, (width, height), interpolation=cv2.INTER_AREA)
        output_image = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
        return output_image, results_detect, list_label_ng
    
class base_handle_video(PLC_Connection,MySQL_Connection):

    def __init__(self):
        self.database = MySQL_Connection(None,None,None,None) 
        self.name_table = None
        self.item_code_cfg = None
        self.image_files = []
        self.current_image_index = 0
        self.state = 0
        self.password = " "
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

    def connect_database(self):
        cursor, db_connection  = self.database.Connect_MySQLServer()
        check_connection = self.database.check_connection()
        reconnect = self.database.reconnect()
        return cursor,db_connection,check_connection,reconnect
    
    def load_data_model_vid(self):
        cursor, db_connection,_,_ = self.connect_database()
        print(self.name_table)
        cursor.execute(f"SELECT * FROM {self.name_table} WHERE item_code = %s", (self.item_code_cfg,))
        records = cursor.fetchall()
        cursor.close()
        db_connection.close()
        if records:
            first_record = records[0]
            load_item_code = first_record['item_code']
            load_path_weight = first_record['weight']
            load_confidence_all_scale = first_record['confidence_all']
        else: 
            first_record = None
            load_item_code = None
            load_path_weight = None
            load_confidence_all_scale = None

        return records,load_path_weight,load_item_code,load_confidence_all_scale

    def load_parameters_model_vid(self,model1,load_path_weight,load_item_code,load_confidence_all_scale,records):
        self.weights.delete(0, tk.END)
        self.weights.insert(0, load_path_weight)
        try:
            self.item_code.delete(0, tk.END)
            self.item_code.insert(0, load_item_code)
            self.scale_conf_all.set(load_confidence_all_scale)
        except: 
            pass
        try:
            for i1 in range(len(model1.names)):          
                for record in records:                
                    if record['label_name'] == model1.names[i1]:
                        self.join[i1].set(bool(record['join_detect']))
                        self.ok_vars[i1].set(bool(record['OK']))
                        self.ng_vars[i1].set(bool(record['NG']))
                        self.num_inputs[i1].delete(0, tk.END)
                        self.num_inputs[i1].insert(0, record['num_labels'])
                        self.wn_inputs[i1].delete(0, tk.END)
                        self.wn_inputs[i1].insert(0, record['width_min'])
                        self.wx_inputs[i1].delete(0, tk.END)
                        self.wx_inputs[i1].insert(0, record['width_max'])
                        self.hn_inputs[i1].delete(0, tk.END)
                        self.hn_inputs[i1].insert(0, record['height_min'])                
                        self.hx_inputs[i1].delete(0, tk.END)
                        self.hx_inputs[i1].insert(0, record['height_max'])
                        self.plc_inputs[i1].delete(0, tk.END)
                        self.plc_inputs[i1].insert(0, record['PLC_value'])
                        self.conf_scales[i1].set(record['cmpnt_conf'])
        except IndexError as e:
            messagebox.showerror("Error", f"Load parameters failed! Error: {str(e)}")


    def select_video(self,path_video):
        selected_file = filedialog.askopenfilename(title="Choose a file", filetypes=[("Video Files Type", "*.mp4")])
        if selected_file:
            path_video.delete(0,tk.END)
            path_video.insert(0,selected_file)

    def render(self,
        progress_label,
        video_canvas,
        source = None,
        device="cpu",
        save_img=True,
        exist_ok=False,
        classes=None,
        line_thickness=2,
    ):
        if source == None : 
            messagebox.showwarning("Warning", "You have to select a valid file")
        vid_frame_count = 0
        # model = YOLO(f"{self.weights.get()}")
        device = '0' if torch.cuda.is_available() else 'cpu'
        self.model.to("cuda") if device == "0" else self.model.to("cpu")
        names = self.model.model.names
        videocapture = cv2.VideoCapture(source)
        frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
        fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")
        total_frames = int(videocapture.get(cv2.CAP_PROP_FRAME_COUNT))
        save_dir = increment_path(Path("ultralytics_source_output") / "exp", exist_ok)
        save_dir.mkdir(parents=True, exist_ok=True)
        video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.mp4"), fourcc, fps, (frame_width, frame_height))
        while videocapture.isOpened():
            success, frame = videocapture.read()
            if not success:
                break
            vid_frame_count += 1
            self.handle_params_video(frame,classes,line_thickness,names)
            # results = self.model.track(frame, persist=True, classes=classes)
            # if results[0].boxes.id is not None:
            #     boxes = results[0].boxes.xyxy.cpu().numpy()
            #     track_ids = results[0].boxes.id.int().cpu().numpy().tolist()
            #     clss = results[0].boxes.cls.cpu().numpy().tolist()
            #     xywhs = results[0].boxes.xywh.cpu().numpy().tolist()
            #     annotator = Annotator(frame, line_width=line_thickness, example=str(names))
            #     for box, track_id, cls, xywh in zip(boxes, track_ids, clss, xywhs):
            #         label = f'{str(track_id)} {names[cls]} x:{str(int(xywh[0]))} y:{str(int(xywh[1]))}'
            #         annotator.box_label(box, label, color=colors(cls, True))
            if save_img:
                video_writer.write(frame)
                self.update_progress(vid_frame_count, total_frames, progress_label)
        video_writer.release()
        videocapture.release()
        cv2.destroyAllWindows()
        self.show(save_dir,video_canvas)

    def update_progress(self, current_frame, total_frames, progress_label):
        device = 'GPU' if torch.cuda.is_available() else 'CPU'
        progress = (current_frame / total_frames) * 100
        progress_label.config(text=f"Rendering on {device} in progress: {progress:.2f}%")
        progress_label.update_idletasks()

    def show(self, save_dir,video_canvas):
        video_path = list(save_dir.glob("*.mp4"))[0]
        cap = cv2.VideoCapture(str(video_path))
        self.update_frame(cap, video_canvas)

    def update_frame(self, cap, video_canvas):
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (600, 550))
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            video_canvas.imgtk = imgtk
            video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        video_canvas.after(20, self.update_frame, cap, video_canvas)

    def start_rendering(self,path_video,progress_label,video_canvas):
        kwargs = {
            "progress_label": progress_label,
            "video_canvas": video_canvas,
            "source": path_video,
            "device": "cpu",
            "save_img": True,
            "exist_ok": False,
            "classes": None,
            "line_thickness": 2
        }
        processing_thread = threading.Thread(target=self.render, kwargs=kwargs)
        processing_thread.start()

    def handle_params_video(self,frame,classes,line_thickness,names):
        results = self.model.track(frame, persist=True, classes=classes)
        model_settings = [
            {
                'label_name':  self.model_name_labels[i1].cget("text"),
                'join_detect': self.join[i1].get(),
                'OK_jont': self.ok_vars[i1].get(),
                'NG_jont': self.ng_vars[i1].get(),
                'num_labels': int(self.num_inputs[i1].get()),
                'width_min': int(self.wn_inputs[i1].get()),
                'width_max': int(self.wx_inputs[i1].get()),
                'height_min': int(self.hn_inputs[i1].get()),
                'height_max': int(self.hx_inputs[i1].get()),
                'PLC_value': int(self.plc_inputs[i1].get()),
                'cmpnt_conf': int(self.conf_scales[i1].get()),
            }
            for i1 in range(len(self.model_name_labels))
        ]
        settings_dict = {setting['label_name']: setting for setting in model_settings}
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy().tolist()
            clss = results[0].boxes.cls.cpu().numpy().tolist()
            xywhs = results[0].boxes.xywh.cpu().numpy().tolist()
            conf_list = results[0].boxes.conf.cpu().numpy().tolist()
            annotator = Annotator(frame, line_width=line_thickness, example=str(names))
            for box, track_id, cls, xywh, conf in zip(boxes, track_ids, clss, xywhs,conf_list):
                label = f'{str(track_id)} {names[cls]} {conf:.2f} x:{str(int(xywh[0]))} y:{str(int(xywh[1]))}'
                if settings_dict[results[0].names[int(cls)]]:
                    if settings_dict[results[0].names[int(cls)]]['join_detect']:
                        if int(conf*100) < settings_dict[results[0].names[int(cls)]]['cmpnt_conf']:
                            annotator.box_label(box, label, color=(0,255,0))
                        else: 
                            annotator.box_label(box, label, color=(255, 0, 0))  

        
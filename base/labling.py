from tkinter import filedialog,messagebox
import os,subprocess

def copy_file_contents(source_path, destination_path):
    try:
        with open(source_path, 'r') as source_file:
            contents = source_file.read()
        with open(destination_path, 'w') as destination_file:
            destination_file.write(contents)
    except: 
        pass
def open_label_img_obb(): 
    source_path = filedialog.askopenfilename(title="Choose a file", filetypes=[("Model Files", "classes.txt")])
    destination_path = os.getcwd() + '/labelImg_OBB/data/predefined_classes.txt'
    copy_file_contents(source_path,destination_path)
    program_dir = os.path.join(os.getcwd(), 'labelImg_OBB' , 'labelImg.py')         
    subprocess.call(['python',program_dir])

def open_label_img_hbb(): 
    source_path = filedialog.askopenfilename(title="Choose a file", filetypes=[("Model Files", "classes.txt")])
    destination_path = os.getcwd() + '/labelImg_OBB/data/predefined_classes.txt'
    copy_file_contents(source_path,destination_path)
    program_dir = os.path.join(os.getcwd(), 'labelImg' , 'labelImg.py')         
    subprocess.call(['python',program_dir])
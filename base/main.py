import tkinter as tk
from model_1 import *
from model_2 import *
from programTraining import *
from videoDetection import *
from tkinter import ttk
from base_model import removefile
from tkinter import *
from tkinter import *
import os,subprocess
from tkinter import filedialog
from tkinter import messagebox,simpledialog
def copy_file_contents(source_path, destination_path):
    try:
        with open(source_path, 'r') as source_file:
            contents = source_file.read()
        with open(destination_path, 'w') as destination_file:
            destination_file.write(contents)
    except: 
        pass
def open_label_img(): 
    source_path = filedialog.askopenfilename(title="Choose a file", filetypes=[("Model Files", "classes.txt")])
    destination_path = os.getcwd() + '/labelImg_OBB/data/predefined_classes.txt'
    copy_file_contents(source_path,destination_path)
    program_dir = os.path.join(os.getcwd(), 'labelImg_OBB' , 'labelImg.py')         
    subprocess.call(['python',program_dir])

def open_tools_window(root):
    tools_window = Toplevel(root)
    tools_window.title("Tools Window")
    tools_window.geometry("800x600")
    label = tk.Label(tools_window, text="This is the Tools window")
    label.pack(pady=20)

def open_setting_window(root):
    setting_window = Toplevel(root)
    setting_window.title("Setting Window")
    setting_window.geometry("800x600")
    label = tk.Label(setting_window, text="This is the Setting window")
    label.pack(pady=20)

def open_crop_window(root):
    crop_window = Toplevel(root)
    crop_window.title("Crop Window")
    crop_window.geometry("800x600")
    label = tk.Label(crop_window, text="This is the Crop window")
    label.pack(pady=20)

def donothing(root):
   filewin = Toplevel(root)
   button = Button(filewin, text="Do nothing button")
   button.pack()

def create_context_menu(notebook):
    context_menu = Menu(notebook, tearoff=0)
    context_menu.add_command(label="Refresh", command=lambda: refresh_tab(notebook))
    context_menu.add_command(label="Hide", command=lambda: hide_tab(notebook))
    context_menu.add_command(label="Close Tab", command=lambda: close_tab(notebook))

    def show_context_menu(event):
        tab_index = notebook.index("@%d,%d" % (event.x, event.y))
        if tab_index != -1:
            notebook.select(tab_index)
            context_menu.post(event.x_root, event.y_root)
    notebook.bind("<Button-3>", show_context_menu)
    return context_menu

def refresh_tab(notebook):
    tab = notebook.select()
    if tab:
        print(f"Refreshing tab: {notebook.tab(tab, 'text')}")

def hide_tab(notebook):
    tab = notebook.select()
    if tab:
        notebook.forget(tab)
        print(f"Hiding tab: {notebook.tab(tab, 'text')}")

def close_tab(notebook):
    tab = notebook.select()
    if tab:
        notebook.forget(tab)
        print(f"Closing tab: {notebook.tab(tab, 'text')}")

def display_layout(notebook, window):
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(window, orient="horizontal", length=200, mode="determinate", variable=progress_var)
    progress_label = tk.Label(window, text="0%")
    loading_label = tk.Label(window, text="Loading model...")

    def update_progress(step, total_steps):
        progress = (step / total_steps) * 100
        progress_var.set(progress)
        progress_label.config(text=f'{int(progress)}%')
        window.update_idletasks()

    total_steps = 10
    step = 0

    progress_bar.pack(pady=10)
    progress_label.pack(pady=10)
    loading_label.pack(pady=10)

    update_progress(step, total_steps)
    
    step += 1
    update_progress(step, total_steps)
    removefile()

    step += 1
    update_progress(step, total_steps)
    display_camera_tab = ttk.Frame(notebook)
    notebook.add(display_camera_tab, text="Display Camera")

    step += 1
    update_progress(step, total_steps)
    tab1 = ttk.Frame(display_camera_tab)
    tab1.pack(side=tk.LEFT, fill="both", expand=True)

    step += 1
    update_progress(step, total_steps)
    tab2 = ttk.Frame(display_camera_tab)
    tab2.pack(side=tk.RIGHT, fill="both", expand=True)

    step += 1
    update_progress(step, total_steps)
    tab_camera_1 = Model_Camera_1()
    display_camera_frame1 = tab_camera_1.Display_Camera(tab1)

    step += 1
    update_progress(step, total_steps)
    tab_camera_1.update_images(window, display_camera_frame1)

    # step += 1
    # update_progress(step, total_steps)
    # tab_camera_2 = Model_Camera_2()
    # display_camera_frame2 = tab_camera_2.Display_Camera(tab2)

    # step += 1
    # update_progress(step, total_steps)
    # tab_camera_2.update_images(window, display_camera_frame2)

    step += 1
    update_progress(step, total_steps)
    settings_notebook = ttk.Notebook(notebook)
    notebook.add(settings_notebook, text="Camera Configure Setup")
    tab_camera_1.Camera_Settings(settings_notebook)
    # tab_camera_2.Camera_Settings(settings_notebook)

    update_progress(total_steps, total_steps)
    progress_var.set(100)
    progress_label.config(text='100%')
    window.update_idletasks()

    progress_bar.pack_forget()
    progress_label.pack_forget()
    loading_label.pack_forget()

def video(notebook, window):
    settings_notebook = ttk.Notebook(notebook)
    notebook.add(settings_notebook, text="Video Detection")
    tab_camera_1 = Video_Dectection()
    tab_camera_1.Video_Settings(settings_notebook)

def training_data(notebook, window):
    settings_notebook = ttk.Notebook(notebook)
    notebook.add(settings_notebook, text="Training")
    tab_camera_1 = Training_Data()
    tab_camera_1.layout(settings_notebook,window)

def confirm_exit(window):
    confirm_exit = messagebox.askokcancel("Confirm", "Are you sure to exit ?")
    if confirm_exit: 
        window.quit() 
    else: 
        pass

def main():
    global menubar
    window = tk.Tk()
    window.title("YOLOv8.2.0 by Utralytics ft Tkinter")
    window.state('zoomed')

    notebook = ttk.Notebook(window)
    notebook.pack(fill="both", expand=True)

    menubar = Menu(window)
    filemenu = Menu(menubar, tearoff=0)
    filemenu.add_command(label="Open Camera Display", command=lambda: display_layout(notebook, window))
    filemenu.add_command(label="Label Image", command=lambda: open_label_img())
    filemenu.add_command(label="Train Datasets", command=lambda: training_data(notebook, window))
    filemenu.add_command(label="Real-Time Integration", command=donothing)
    filemenu.add_command(label="Extract Output", command=lambda: video(notebook, window))
    filemenu.add_separator()
    filemenu.add_command(label="Exit", command=lambda:confirm_exit(window))
    menubar.add_cascade(label="Tools", menu=filemenu)

    helpmenu = Menu(menubar, tearoff=0)
    helpmenu.add_command(label="About...", command=donothing)
    menubar.add_cascade(label="Help", menu=helpmenu)

    window.config(menu=menubar)
    create_context_menu(notebook)
    window.mainloop()

if __name__ == "__main__":
    main()
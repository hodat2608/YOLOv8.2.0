import tkinter as tk
from model_1 import *
from base.train import *
from base.extract_vid import *
from tkinter import ttk
from base.ultils import removefile
from tkinter import *
from tkinter import messagebox
from base.labling import *

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

    # step += 1
    # update_progress(step, total_steps)
    # tab2 = ttk.Frame(display_camera_tab)
    # tab2.pack(side=tk.RIGHT, fill="both", expand=True)

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

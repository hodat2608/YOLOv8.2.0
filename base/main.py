# import sys
# from pathlib import Path
# current_dir = Path(__file__).resolve().parent.parent
# ultralytics_main_dir = current_dir
# sys.path.append(str(ultralytics_main_dir))
import root_path
import tkinter as tk
# from model_1 import *
from base.train import *
from base.extract_vid import *
from tkinter import ttk
from base.ultils import removefile
from tkinter import *
from tkinter import *
import os,subprocess
from tkinter import filedialog
from tkinter import messagebox
# from base.menu import *
from base.labling import *
from base.menu_cfg import *
from base.model_1_config import *

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
    open_label_img_menu(filemenu)
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
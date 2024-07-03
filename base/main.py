import tkinter as tk
from model_1 import Model_Camera_1
from model_2 import Model_Camera_2
from tkinter import ttk, filedialog
from base_model import removefile

def main():
    window = tk.Tk()
    window.title("YOLOv8.2.0 by Utralytics ft Tkinter")
    window.state('zoomed')

    notebook = ttk.Notebook(window)
    notebook.pack(fill="both", expand=True)
    removefile()

    display_camera_tab = ttk.Frame(notebook)
    notebook.add(display_camera_tab, text="Display Camera")

    tab1 = ttk.Frame(display_camera_tab)
    tab1.pack(side=tk.LEFT, fill="both", expand=True)

    tab2 = ttk.Frame(display_camera_tab)
    tab2.pack(side=tk.RIGHT, fill="both", expand=True)

    tab_camera_1 = Model_Camera_1()
    display_camera_frame1 = tab_camera_1.Display_Camera(tab1)
    tab_camera_1.update_images(window, display_camera_frame1)

    tab_camera_2 = Model_Camera_2()
    display_camera_frame2 = tab_camera_2.Display_Camera(tab2)
    tab_camera_2.update_images(window, display_camera_frame2)

    settings_notebook = ttk.Notebook(notebook)
    notebook.add(settings_notebook, text="Camera Configure Setup")
    tab_camera_1.Camera_Settings(settings_notebook)
    tab_camera_2.Camera_Settings(settings_notebook)

    window.mainloop()

if __name__ == "__main__":
    main()
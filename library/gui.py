import os
import sys

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.font as tkfont
import tkinter.messagebox as tkmessagebox

from library.gui_utils import AnalysesWidget


class MorphometricAnalyzerGUI(ttk.Frame):

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)

        self.images = {}

        self.create_top_frame()
        self.create_parameters_frame()
        ttk.Separator(self, orient="horizontal").grid(
            row=1, column=0, columnspan=3, sticky="we")
        self.grid(row=0, column=0, sticky="nsew")

    def create_top_frame(self) -> None:

        top_frame = ttk.Frame(self)
        top_frame.columnconfigure(6, weight=1)
        top_frame.rowconfigure(0, weight=1)
        top_frame.grid(row=0, column=0, columnspan=3)

        ttk.Label(top_frame, text="Morphometricanalyzer",
                  font=tkfont.Font(size=20)).grid(row=0, column=0)
        ttk.Separator(top_frame, orient="vertical").grid(
            row=0, column=1, sticky="nsew")

        for image_key, image_file, text, column in (
                ("open_button", "open.png", "open", 2),
                ("save_button", "save.png", "save", 3),
                ("save_all_button", "save_all.png", "save_all", 4),
                ("run_button", "run.png", "run", 5)):
            self.images[image_key] = tk.PhotoImage(
                file=os.path.join(sys.path[0], "data", image_file))
            ttk.Button(top_frame, text=text,
                       image=self.images[image_key], compound="top", style="Toolbutton", padding=(10, 0)).grid(row=0, column=column)

        ttk.Label(top_frame).grid(row=0, column=6)
        ttk.Separator(top_frame, orient="vertical").grid(
            row=0, column=7, sticky="nsew")
        self.images["logo"] = tk.PhotoImage(file=os.path.join(
            sys.path[0], "data", "iTaxoTools Digital linneaeus MICROLOGO.png"))
        ttk.Label(top_frame, image=self.images["logo"]).grid(
            row=0, column=8, sticky="nse")

    def create_parameters_frame(self) -> None:
        parameters_frame = ttk.LabelFrame(self, text="Parameters")
        parameters_frame.grid(row=3, column=0, sticky="nsew")

        ttk.Label(
            parameters_frame, text="Number of analyses").grid(row=0, column=0, columnspan=2)

        self.num_anylyses = tk.StringVar()
        ttk.Entry(
            parameters_frame, textvariable=self.num_anylyses).grid(row=1, column=0)

        ttk.Button(
            parameters_frame, text="Set", command=self.set_num_analyses).grid(row=1, column=1)

        self.analyses_widget = AnalysesWidget(parameters_frame)
        self.analyses_widget.grid(row=2, column=0, columnspan=2)
        self.analyses_widget.set_count(3)
        self.analyses_widget.frame.configure(relief="groove", padding=3)

    def set_num_analyses(self) -> None:
        try:
            num = int(self.num_anylyses.get())
        except ValueError:
            tkmessagebox.showwarning(
                title="Warning", message=f"Can't set number of analyses to {self.num_anylyses.get()}")
            return
        else:
            self.analyses_widget.set_count(num)

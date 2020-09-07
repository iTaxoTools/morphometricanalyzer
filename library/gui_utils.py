import os.path
import tkinter as tk
import tkinter.filedialog
import tkinter.ttk as ttk
from typing import Literal, Any


class FileChooser():
    """
    Creates a frame with a label, entry and browse button for choosing files
    """

    def __init__(self, parent: Any, *, label: str, mode: Literal["open", "save"]):
        self.frame = ttk.Frame(parent)
        self.frame.columnconfigure([0, 1], weight=1)
        self.label = ttk.Label(self.frame, text=label)
        self.file_var = tk.StringVar()
        self.entry = ttk.Entry(self.frame, textvariable=self.file_var)
        if mode == "open":
            self._dialog = tk.filedialog.askopenfilename
        elif mode == "save":
            self._dialog = tk.filedialog.asksaveasfilename

        def browse() -> None:
            self.file_var.set(os.path.relpath(self._dialog()))

        self.button = ttk.Button(self.frame, text="Browse", command=browse)

        self.label.grid(row=0, column=0, sticky='nws')
        self.entry.grid(row=1, column=0, sticky='nwse')
        self.button.grid(row=1, column=1)
        self.grid = self.frame.grid

import os.path
import tkinter as tk
import tkinter.filedialog as tkfiledialog
import tkinter.ttk as ttk
from typing import Any, Dict, Tuple, List, Optional


class FileChooser():
    """
    Creates a frame with a label, entry and browse button for choosing files
    """

    def __init__(self, parent: Any, *, label: str, mode: str):
        self.frame = ttk.Frame(parent)
        self.frame.columnconfigure([0, 1], weight=1)
        self.label = ttk.Label(self.frame, text=label)
        self.file_var = tk.StringVar()
        self.entry = ttk.Entry(self.frame, textvariable=self.file_var)
        if mode == "open":
            self._dialog = tkfiledialog.askopenfilename
        elif mode == "save":
            self._dialog = tkfiledialog.asksaveasfilename
        elif mode == "dir":
            self._dialog = tkfiledialog.askdirectory

        def browse() -> None:
            newpath: Optional[str] = self._dialog()
            if newpath:
                try:
                    newpath = os.path.relpath(newpath)
                except:
                    newpath = os.path.abspath(newpath)
                self.file_var.set(newpath)

        self.button = ttk.Button(self.frame, text="Browse", command=browse)

        self.label.grid(row=0, column=0, sticky='nws')
        self.entry.grid(row=1, column=0, sticky='nwse')
        self.button.grid(row=1, column=1)
        self.grid = self.frame.grid


class AnalysisOptions():
    """
    Class that constists of a label and checkboxes 'sex', 'species', 'locality'
    """
    default_choices = {'species', 'sex'}

    def __init__(self, parent: tk.Widget, *, label: str):
        self.frame = ttk.Frame(parent)
        self.label = ttk.Label(self.frame, text=label)

        # create checkoboxes
        self.checkboxes: Dict[str, Tuple[tk.BooleanVar, ttk.Checkbutton]] = {}
        for name in ['species', 'sex', 'locality']:
            var = tk.BooleanVar()
            if name in AnalysisOptions.default_choices:
                var.set(True)
            self.checkboxes[name] = (var, ttk.Checkbutton(
                self.frame, text=name.capitalize(), variable=var))

        # set the invoke command for checkboxes
        for _, chkbox in self.checkboxes.values():
            chkbox.configure(command=self.set_minimal)

        # grid the subwidgets
        self.label.grid(row=0, column=0)
        for i, (_, chkbox) in enumerate(self.checkboxes.values()):
            chkbox.grid(row=0, column=(1+i))

        # delegate grid and destroy
        self.grid = self.frame.grid
        self.destroy = self.frame.destroy

    def get(self) -> List[str]:
        """
        returns list of choices
        """
        return [name for name, (var, _) in self.checkboxes.items() if var.get()]

    def set_minimal(self) -> None:
        """
        Sets species, if everything is unset
        """
        if all(not val.get() for (val, _) in self.checkboxes.values()):
            self.checkboxes['species'][0].set(True)


class AnalysesWidget():
    """
    Class for containing a group of AnalysisOptions widgets

    Can change the number of widgets inside
    """

    def __init__(self, parent: tk.Widget):
        self.frame = ttk.Frame(parent)
        self.children: List[AnalysisOptions] = []
        self.grid = self.frame.grid

    def set_count(self, n: int) -> None:
        """
        Creates or destroy AnalysisOptions widgets, so that their count becomes n
        """
        n = max(n, 0)
        current_count = len(self.children)
        if n < current_count:  # destroy children if there are too much
            for child in self.children[n:]:
                child.destroy()
            self.children = self.children[0:n]  # perhaps del self.children[n:]
        elif n > current_count:  # create children, if there are not enough
            for i in range(current_count, n):
                self.children.append(AnalysisOptions(
                    self.frame, label=f"Analysis {i}: Separate by "))
                self.children[i].grid(row=i, column=0)

    def get(self) -> List[List[str]]:
        return [child.get() for child in self.children]


class LabeledEntry():
    """
    Group of a label, entry and a string variable
    """

    def __init__(self, parent: tk.Widget, *, label: str):
        self.frame = tk.Frame(parent)
        self.label = tk.Label(self.frame, text=label)
        self.var = tk.StringVar()
        self.entry = tk.Entry(self.frame, textvariable=self.var)
        self.frame.columnconfigure(1, weight=1)
        self.label.grid(column=0, row=0)
        self.entry.grid(column=1, row=0)
        self.grid = self.frame.grid

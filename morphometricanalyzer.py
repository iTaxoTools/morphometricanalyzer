#!/usr/bin/env python3
import sys
import tkinter as tk
import tkinter.messagebox
import tkinter.ttk as ttk

from library.gui_utils import *
from library.mistake_corrector import *


def gui_main() -> None:
    root = tk.Tk()
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)
    mainframe = ttk.Frame(root, padding=5)
    mainframe.columnconfigure([0, 1, 2], weight=1)

    input_chooser = FileChooser(mainframe, label="Input file", mode="open")
    output_chooser = FileChooser(mainframe, label="Output file", mode="save")
    table_chooser = FileChooser(
        mainframe, label="Modified table file", mode="save")

    def process_table() -> None:
        input_file = input_chooser.file_var.get()
        output_file = output_chooser.file_var.get()
        table_file = table_chooser.file_var.get()

        try:
            with open(input_file) as input_file, open(output_file, mode='w') as output_file, open(table_file, mode='w') as table_file:
                corrector = MistakeCorrector(input_file)
                for line in corrector:
                    print(line, file=table_file)
                corrector.report(output_file)
        except ValueError as ex:
            tk.messagebox.showerror("Error", str(ex))
        except FileNotFoundError as ex:
            if ex.filename:
                tk.messagebox.showerror("Error", str(ex))
            else:
                tk.messagebox.showerror(
                    "Error", "One of the file names is empty.")

    process_btn = ttk.Button(mainframe, text="Process", command=process_table)

    input_chooser.grid(row=0, column=0, sticky="nsew")
    output_chooser.grid(row=0, column=1, sticky="nsew")
    table_chooser.grid(row=0, column=2, sticky="nsew")
    process_btn.grid(row=1, column=1)

    mainframe.grid(row=0, column=0, sticky="nsew")

    root.mainloop()


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--cmd":
        corrector = MistakeCorrector(sys.stdin)
        for line in corrector:
            print(line)
        corrector.report(sys.stderr)
    else:
        gui_main()


if __name__ == "__main__":
    main()

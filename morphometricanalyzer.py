#!/usr/bin/env python3
import sys
import tkinter as tk
import tkinter.messagebox
import tkinter.ttk as ttk
import io
import warnings

from library.gui_utils import *
from library.mistake_corrector import *
from library.analyses import analyse


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

    analyses_widget = AnalysesWidget(mainframe)

    def process_table() -> None:
        input_file = input_chooser.file_var.get()
        output_file = output_chooser.file_var.get()
        table_file = table_chooser.file_var.get()

        try:
            with open(input_file) as input_file, open(output_file, mode='w') as output_file, open(table_file, mode='w') as table_file:
                corrector = MistakeCorrector(input_file)
                buf = io.StringIO()
                for line in corrector:
                    print(line, file=table_file)
                    print(line, file=buf)
                corrector.report(output_file)
                buf.seek(0, 0)
                with warnings.catch_warnings(record=True) as warns:
                    analyse(buf, output_file,
                            corrector.header_fixer.variables, analyses_widget.get())
                    for w in warns:
                        tk.messagebox.showwarning("Warning", str(w.message))
                    tk.messagebox.showinfo("Done", "All analyses are complete")
        except ValueError as ex:
            tk.messagebox.showerror("Error", str(ex))
        except FileNotFoundError as ex:
            if ex.filename:
                tk.messagebox.showerror("Error", str(ex))
            else:
                tk.messagebox.showerror(
                    "Error", "One of the file names is empty.")

    process_btn = ttk.Button(mainframe, text="Process", command=process_table)

    set_num_analyses_frm = ttk.Frame(mainframe)
    num_analyses_lbl = ttk.Label(
        set_num_analyses_frm, text="Number of analyses")
    num_analyses_var = tk.StringVar()
    num_analyses_entr = ttk.Entry(
        set_num_analyses_frm, textvariable=num_analyses_var)

    def set_num_analyses() -> None:
        try:
            num = int(num_analyses_var.get())
        except ValueError:
            return
        else:
            analyses_widget.set_count(num)

    num_analyses_btn = ttk.Button(
        set_num_analyses_frm, text="Set", command=set_num_analyses)

    input_chooser.grid(row=0, column=0, sticky="nsew")
    output_chooser.grid(row=0, column=1, sticky="nsew")
    table_chooser.grid(row=0, column=2, sticky="nsew")
    process_btn.grid(row=1, column=1)

    num_analyses_lbl.grid(row=0, column=0)
    num_analyses_entr.grid(row=0, column=1)
    num_analyses_btn.grid(row=0, column=2)
    set_num_analyses_frm.grid(row=2, column=0, columnspan=3, sticky='w')

    analyses_widget.grid(row=3, column=0, columnspan=3, sticky='w')

    mainframe.grid(row=0, column=0, sticky="nsew")

    root.mainloop()


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--cmd":
        corrector = MistakeCorrector(sys.stdin)
        buf = io.StringIO()
        for line in corrector:
            print(line, file=buf)
            print(line)
        corrector.report(sys.stderr)
        buf.seek(0, 0)
        with warnings.catch_warnings(record=True) as warns:
            analyse(buf, sys.stderr,
                    corrector.header_fixer.variables, [['species', 'sex']])
            for w in warns:
                print(w.message, file=sys.stderr)
    else:
        gui_main()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from library.analyses import Analyzer
from library.mistake_corrector import *
from library.gui_utils import *
import sys
import tkinter as tk
import tkinter.messagebox as tkmessagebox
import tkinter.ttk as ttk
import io
import os
import warnings
from datetime import datetime, timezone
import logging

try:
    os.mkdir(os.path.join(sys.path[0], "logs"))
except Exception:
    pass

log_file_name = f"log_{datetime.now(timezone.utc):%d-%b-%Y_%H-%M-%S}UTC.txt"
logging.basicConfig(filename=os.path.join(
    sys.path[0], "logs", log_file_name), level=logging.INFO)


def gui_main() -> None:
    root = tk.Tk()
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)
    mainframe = ttk.Frame(root, padding=5)
    mainframe.columnconfigure([0, 1], weight=1)

    input_chooser = FileChooser(mainframe, label="Input file", mode="open")
    output_chooser = FileChooser(
        mainframe, label="Output directory", mode="dir")

    analyses_widget = AnalysesWidget(mainframe)

    size_var_chooser = LabeledEntry(
        mainframe, label="Variable to be used for size standardization")

    def process_table() -> None:
        input_file = input_chooser.file_var.get()
        output_dir = output_chooser.file_var.get()
        output_file = os.path.join(output_dir, "output.txt")
        table_file = os.path.join(output_dir, "table.txt")
        logging.info(f"Processing, input: {input_file}, output: {output_dir}")

        try:
            with open(input_file, errors='replace') as input_file, open(output_file, mode='w') as output_file, open(table_file, mode='w') as table_file:
                corrector = MistakeCorrector(input_file)
                buf = io.StringIO()
                for line in corrector:
                    print(line, file=buf)
                corrector.report(output_file)
                output_file.write("\n\n\n")
                buf.seek(0, 0)
                analyzer = Analyzer(buf, corrector.header_fixer.variables, [
                    ['species', 'sex']], output_file, table_file)
                analyzer.set_size_var(size_var_chooser.var.get().casefold())
                with warnings.catch_warnings(record=True) as warns:
                    analyzer.analyse()
                    tkmessagebox.showwarning("Warning", '\n\n'.join(
                        set(str(w.message) for w in warns)))
                    tkmessagebox.showinfo("Done", "All analyses are complete")
                    logging.info("Processing successful\n")
        except FileNotFoundError as ex:
            logging.error(ex)
            if ex.filename:
                tkmessagebox.showerror("Error", str(ex))
            else:
                tkmessagebox.showerror(
                    "Error", "One of the file names is empty.")
        except Exception as ex:
            logging.error(ex)
            tkmessagebox.showerror("Error", str(ex))

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
            tkmessagebox.showwarning(
                title="Warning", message="Can't set number of analyses to {num_analyses_var.get()}")
            return
        else:
            analyses_widget.set_count(num)

    num_analyses_btn = ttk.Button(
        set_num_analyses_frm, text="Set", command=set_num_analyses)

    input_chooser.grid(row=0, column=0, sticky="nsew")
    output_chooser.grid(row=0, column=1, sticky="nsew")
    process_btn.grid(row=1, column=0, columnspan=2)

    size_var_chooser.grid(row=2, column=0, columnspan=3, sticky='w')
    num_analyses_lbl.grid(row=0, column=0)
    num_analyses_entr.grid(row=0, column=1)
    num_analyses_btn.grid(row=0, column=2)
    set_num_analyses_frm.grid(row=3, column=0, columnspan=3, sticky='w')

    analyses_widget.grid(row=4, column=0, columnspan=3, sticky='w')

    mainframe.grid(row=0, column=0, sticky="nsew")

    root.mainloop()


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--cmd":
        corrector = MistakeCorrector(sys.stdin)
        buf = io.StringIO()
        for line in corrector:
            print(line, file=buf)
        corrector.report(sys.stderr)
        sys.stderr.write("\n\n\n")
        buf.seek(0, 0)
        analyzer = Analyzer(buf, corrector.header_fixer.variables, [
                            ['species', 'sex']], sys.stderr, sys.stdout)
        with warnings.catch_warnings(record=True) as warns:
            analyzer.analyse()
            for message in set(str(w.message) for w in warns):
                print(message, file=sys.stderr)
    else:
        gui_main()


if __name__ == "__main__":
    main()

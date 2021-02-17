#!/usr/bin/env python3
from library.gui import MorphometricAnalyzerGUI
from library.mistake_corrector import MistakeCorrector
from library.analyses import Analyzer
import sys
import tkinter as tk
import io
import os
import warnings
from datetime import datetime, timezone
import logging
import tempfile

try:
    os.mkdir(os.path.join(sys.path[0], "logs"))
except Exception:
    pass

log_file_name = f"log_{datetime.now(timezone.utc):%d-%b-%Y_%H-%M}UTC.log"
logging.basicConfig(filename=os.path.join(
    sys.path[0], "logs", log_file_name), level=logging.INFO)


def gui_main() -> None:
    root = tk.Tk()

    def close_window():
        root.destroy()
        root.quit()

    root.protocol("WM_DELETE_WINDOW", close_window)
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)

    preview_dir = tempfile.mkdtemp()
    MorphometricAnalyzerGUI(root, preview_dir=preview_dir)

    root.mainloop()
    root.quit()


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
                            ['species', 'sex']], sys.stderr, sys.stdout, tempfile.mkdtemp())
        with warnings.catch_warnings(record=True) as warns:
            analyzer.analyse()
            for message in set(str(w.message) for w in warns):
                print(message, file=sys.stderr)
    else:
        gui_main()


if __name__ == "__main__":
    main()

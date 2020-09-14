# morphometricanalyzer

Performs corrections and statistical analyses on tab-separated files of morphometric data

## Usage
```
morphometricanalyzer.py --cmd < input_file > table_file 2> output_file

morphometricanalyzer.py

    input_file    file with the input data
    table_file    file to write the corrected input data
    output_file   file to report the correction and write the results of analyses
```

The program is provided with three files: input file, output file and modified table file. It reads data from the input file,
which should be tab-separated data file with a header. To the modified table file it writes the corrected input data. To the
output file it writes the messages about the performed corrections and the results of statistical analyses. Currently,
no statistical analyses is performed.

## Dependencies
* pandas
* statsmodels

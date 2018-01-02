#! /usr/bin/env python
"""
Compares output of runs from matlab and python code variants.

To add a new run create a new directory in artifacts/

    <run_name>/
        matlab/
        python/

Files within matlab/ and python/ will be directly compared.

For each csv within matlab/ there must be an equivalent csv in python/ which
has the same name plus the prefix 'python - '.
"""
import csv
from numpy import allclose
from os import path, listdir


def main():
    base_path = path.join('artifacts')

    artifact_dirs = [dirName for dirName in listdir(base_path) if
                     path.isdir(path.join(base_path, dirName))]
    for run_dir in artifact_dirs:
        run_path = path.join(base_path, run_dir)
        matlab_run_path = path.join(run_path, 'matlab')
        python_run_path = path.join(run_path, 'python')
        matlab_files = listdir(matlab_run_path)

        print '-' * 50
        print 'Comapring output from run: ' + run_path

        for filename in matlab_files:
            matlab_filepath = path.join(matlab_run_path, filename)
            python_filename = 'python - ' + filename
            python_filepath = path.join(python_run_path, python_filename)
            matlab_rows = None
            with open(matlab_filepath, 'rb') as m_fh:
                reader = csv.reader(m_fh, quoting=csv.QUOTE_NONNUMERIC)
                matlab_rows = list(reader)

            python_rows = None
            with open(python_filepath, 'rb') as p_fh:
                reader = csv.reader(p_fh, quoting=csv.QUOTE_NONNUMERIC)
                python_rows = list(reader)
            print "Comparing: "
            print matlab_filepath
            print python_filepath
            print "Values within absolute tolerance of 1e-13: "
            print allclose(matlab_rows, python_rows, rtol=0.0, atol=1e-13)

if __name__ == '__main__':
    main()
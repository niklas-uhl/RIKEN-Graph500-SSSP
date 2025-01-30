#!/usr/bin/env python3

import sys
import re
import os
# error handling
if len(sys.argv) != 2:
    print("Usage: make_csv.py <directory>")
    sys.exit(1)
    # get first arg as directory to search for files
directory = sys.argv[1]

data = {}

# iterate over all files ending with -log.txt
for file in os.listdir(directory):
    if file.endswith("-log.txt"):

        # open file and read lines
        with open(os.path.join(directory, file), "r") as f:
            if file.endswith("-error-log.txt"):
                kagen_option_string = re.search(r"Using Kagen with option string: (.*)", f.read())
                if kagen_option_string:
                    kagen_option_string = kagen_option_string.group(1)
                    kagen_option_string = {x.split("=")[0]: x.split("=")[1] for x in kagen_option_string.split(";")}
                    kagen_option_string['n'] =int(kagen_option_string['n'])
                    kagen_option_string['m'] =int(kagen_option_string['m'])
                    file = file.strip("-error-log.txt")
                    if file in data:
                        data[file] = (data[file][0], data[file][1], kagen_option_string)
                    else:
                        data[file] = (None, None, kagen_option_string)
            else:
                # get line "num_mpi_processes: *" and extract number
                p = re.search(r"num_mpi_processes:\s*(\d+)", f.read())
                if p:
                    p = int(p.group(1))
                # find mean_time as float
                f.seek(0)
                time = re.search(r"mean_time:\s*([\d.]+)", f.read())
                if time:
                    time = float(time.group(1))
                if p and time:
                        file = file.strip("-log.txt")
                        if file in data:
                                data[file] = (p, time, data[file][2])
                        else:
                                data[file] = (p, time, None)
            # print data as csv
            if file in data and data[file][0] and data[file][1] and data[file][2]:
                p, time, kagen_option_string = data[file]
                kagen_option_string['n'] = int(int(kagen_option_string['n']) / p)
                kagen_option_string['m'] = int(int(kagen_option_string['m']) / p)
                kagen_option_string = f"{kagen_option_string['type']};n={kagen_option_string['n']};m={kagen_option_string['m']}"
                print(f"{kagen_option_string};{p};{time}")
                del data[file]                                       


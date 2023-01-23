#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os


filename_sequential = "sssp_seqential.txt"
filename_parallel_prefix = "sssp_parallel"
sizes = [10, 11, 12, 13, 15, 17, 18, 19]
n_processes = [28, 64]

def writeSequential(n, root):
    print("starting sequential run n=" + str(n) + " root=" + str(root) )
    cmd = "../build/bin/sssp-sequential instances/graph" + str(n) + ".txt " +  str(root) + " > " + filename_sequential
    os.system(cmd) # returns the exit status
# ------------------------

    
def writeParallel(n, np):
    print("starting parallel run n=" + str(n) + " np=" + str(np) )
    cmd = "mpirun -np " + str(np) + " ../build/bin/sssp-parallel " + str(n) + " > /dev/null 2>&1"
    os.system(cmd) # returns the exit status
# ------------------------


def getNfails():
    cmd = "python3 checkResults.py "  +  filename_sequential + " " + filename_parallel_prefix + "*.txt" + " > tmp.txt"
    os.system(cmd)
    
    with open('tmp.txt', 'r') as f:
        last_line = f.readlines()[-1]
        
    nfails = int(last_line.split(':')[1])
    print("nfails in single call: " + str(nfails))

    return nfails
# ------------------------

def getRoot():
    files_list = os.listdir()
    file_name = ''
    for f in files_list:
        if filename_parallel_prefix in f:
            file_name = f
            break
    root = file_name.split('_')[2].split('.')[0]
    return int(root)
    # ------------------------


def resultsAreConsistent(np):
    areConsistent = True
    
    for i in range(0, len(sizes)):
        print("checking n=" + str(sizes[i]) + " np=" + str(np))
        writeParallel(sizes[i], np)
        root = getRoot() + 1       
        writeSequential(sizes[i], root)
        
        nfails = getNfails()

        if nfails > 0:
            areConsistent = False
            break
        cmd = "rm " + filename_sequential + " tmp.txt; rm " + filename_parallel_prefix + "*.txt" 
        os.system(cmd)
    
    return areConsistent
# ------------------------

def main():

    if len(sys.argv) != 1:
        print('usage: test.py')
        return 0

    n_fails = 0
    for np in n_processes:
        if not resultsAreConsistent(np):
            n_fails += 1
            break
    
    print("finished, number of failed runs: " + str(n_fails))
# ------------------------


if __name__ == "__main__":
	sys.exit( main() )
# ------------------------


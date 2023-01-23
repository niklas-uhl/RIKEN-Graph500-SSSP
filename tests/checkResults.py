#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import math

p_digits = 3
eps = 2e-5
inf = 1e10

def getIndexDistPair(line):
    tokens = line.split('=')
    index = (tokens[0])[5:-1]
    dist = tokens[1]
    
    return int(index), float(dist)


def process(filename_ordered, filename_unordered):
    file_ordered = open(filename_ordered, 'r')
    file_unordered = open(filename_unordered, 'r')
    lines_ordered = file_ordered.readlines()
    lines_unordered = file_unordered.readlines()
    n_unordered = 0
    n_ordered = 0
    n_errors = 0
        
    # check for each parallel file entry whether there is a corresponding entry in the sequential file
    for i in range(0, len(lines_unordered)):
        line = lines_unordered[i]

        if not line.startswith("dist"):
            continue
        
        index, dist = getIndexDistPair(line)
        index_ordered, dist_ordered = getIndexDistPair(lines_ordered[index - 1])
        
        n_unordered += 1
        
        if index != index_ordered:
            n_errors += 1
            print("error for unordered line" + str(i));
            
        if round(dist, p_digits) != round(dist_ordered, p_digits) and abs(dist - dist_ordered) > eps:
            n_errors += 1
            print("dist error for unordered line: " + str(dist) + " vs " + str(dist_ordered))
            print("dist error(rounded) for unordered line: " + str(round(dist, p_digits)) + " vs " + str(round(dist_ordered, p_digits)))


    # check that both files have the same number of visited entries
    for i in range(0, len(lines_ordered)):
        line = lines_ordered[i]

        if not line.startswith("dist"):
            continue
        
        index, dist = getIndexDistPair(line)
        
        if dist < inf:
            n_ordered += 1
        
    if n_ordered != n_unordered:
        n_errors += 1
        print("different number of reached vertices: " + str(n_unordered) + " != " + str(n_ordered))
        
    file_ordered.close()
    file_unordered.close()

    return n_errors

def main():

    if len(sys.argv) != 3:
        print('usage: checkResults.py seq_results.txt parallel_results.txt')
        return 0

    n_errors = process(sys.argv[1], sys.argv[2])
    print("finished, number of errors: " + str(n_errors))

if __name__ == "__main__":
	sys.exit( main() )

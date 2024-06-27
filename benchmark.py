from itertools import permutations, product
from math import factorial
from pathlib import Path

import os
import timeit

import numpy as np

from score_strokes import alignStrokes, greedyAlign2, strokeErrorMatrix
from xmlparse import extractBases, loadGeometryBases, loadRef, getXmlScore, minXml

## I can't stop Jupyter Notebook from printing out the genome fitness for every single trial so it's better to put the code in a Python script.
## Simple benchmark test for gene scoring algorithms.

"""
Exhaustive search
"""
def exhaustive():
    ref_geometry, ref_progress_percentage, output_size = ref_data
    g_data, han_chars, base_data, stroke_sets, _, f_names = char_data

    exhaust_scores = {}
    for (geometry_length, bases, stroke_set, f_name) in zip(g_data, base_data, stroke_sets, f_names):
        exhaust_maxes = []
        exhaustive_alignments = permutations(range(1, len(ref_geometry)+1))
        exhaustive_scores = np.zeros(factorial(len(ref_geometry)))
        for j, p in enumerate(exhaustive_alignments):
            p_xml = minXml(ref_char, bases, stroke_set, p)
            exhaustive_scores[j] = getXmlScore(p_xml)
        exhaust_maxes.append(exhaustive_scores.max())
        exhaust_scores[f_name] = np.max(exhaust_maxes)
    return exhaust_scores
    
"""
Holiday's original greedy algorithm
"""
def greedy():
    ref_geometry, ref_progress_percentage, output_size = ref_data
    g_data, _, base_data, stroke_sets, _, f_names = char_data
    #heuristic_scores = []
    heuristic_scores = {}
    for (geometry_length, bases, stroke_set, f_name) in zip(g_data, base_data, stroke_sets, f_names):
        geometry, progress_percentage = geometry_length
        heuristic_alignment = np.array(alignStrokes(geometry, ref_geometry, progress_percentage, ref_progress_percentage))+1
        heuristic_xml = minXml(ref_char, bases, stroke_set, heuristic_alignment)
        heuristic_score = getXmlScore(heuristic_xml)
        #heuristic_scores.append(heuristic_score)
        heuristic_scores[f_name] = heuristic_score
    return heuristic_scores

"""
First heuristic algorithm: O(n^2)

Methodology: Iterate over rows and columns of the error matrix and build a stroke order for each based on the smallest error
"""
def first_heuristic():
    ref_geometry, ref_progress_percentage, output_size = ref_data
    g_data, _, base_data, stroke_sets, _, f_names = char_data
    heuristic_scores = {}#[]
    for (geometry_length, bases, stroke_set, f_name) in zip(g_data, base_data, stroke_sets, f_names): # Iterating over gene data
        #print(f_name)
        strokes, p_strokes = geometry_length
        base_matrix = strokeErrorMatrix(strokes, ref_geometry, p_strokes, ref_progress_percentage)
        #if len(ref_geometry) != len(strokes):
        #    print("skip", f_name)
        #    continue
        #print(base_matrix)
        error_maps = np.copy(base_matrix)
        row_stroke_map = np.full(len(strokes), -1)
        col_stroke_map = np.full(len(strokes), -1)
        row_mins = np.min(error_maps, axis=1)
        col_mins = np.min(error_maps, axis=0)
        compare_scores = []
        stroke_maps = {}
        # Iterate over every smallest error per row
        for row_min in range(len(ref_geometry)):
            coords = np.argwhere(error_maps == row_mins[row_min])
            if len(coords) > 1: # In cases where there are identical error values
                for coord in coords:
                    if not np.any(row_stroke_map == coord[0]):#row_stroke_map[coord[0]-1] != -1:
                        loc = coord
                        break
            else:
                loc = coords[0] # Find [row, col] index of current smallest error
            while row_stroke_map[loc[1]] != -1: # Make sure there's no overlap
                error_maps[loc[0]][loc[1]] = 10000
                loc[1] = np.argmin(error_maps[loc[0]])
            row_stroke_map[loc[1]] = loc[0]
            #print(row_stroke_map)
        if np.array2string(row_stroke_map) not in stroke_maps:
            stroke_maps[np.array2string(row_stroke_map)] = row_stroke_map
        error_maps = np.copy(base_matrix)
        # Iterate by column instead now
        for col_min in range(len(ref_geometry)):
            coords = np.argwhere(error_maps == col_mins[col_min])
            if len(coords) > 1: # In cases where there are identical error values
                for coord in coords:
                    if col_stroke_map[coord[1]-1] != -1:
                        loc = coord
                        break
            else:
                loc = coords[0] # Find [row, col] index of current smallest error
            while np.any(col_stroke_map == loc[0]): # Make sure there's no overlap
                error_maps[loc[0]][loc[1]] = 10000
                loc[0] = np.argmin(error_maps[:, loc[1]])
            col_stroke_map[loc[1]] = loc[0]
        if np.array2string(col_stroke_map) not in stroke_maps:
            stroke_maps[np.array2string(col_stroke_map)] = col_stroke_map
        for s in stroke_maps.values():
            #print(s)
            heuristic_alignment = np.delete(s, np.where(s == -1))+1
            heuristic_xml = minXml(ref_char, bases, stroke_set, heuristic_alignment)
            try:
                heuristic_score = getXmlScore(heuristic_xml)
            except:
                print("err:", f_name)
            compare_scores.append(heuristic_score)
        #heuristic_scores.append(max(compare_scores))
        heuristic_scores[f_name] = max(compare_scores)
    return heuristic_scores

"""
First heuristic algorithm using the greedy algorithm as a fallback
"""
def heuristic_fallback():
    ref_geometry, ref_progress_percentage, output_size = ref_data
    g_data, _, base_data, stroke_sets, _, f_names = char_data
    heuristic_scores = {}#[]
    for (geometry_length, bases, stroke_set, f_name) in zip(g_data, base_data, stroke_sets, f_names): # Iterating over gene data
        #print(f_name)
        strokes, p_strokes = geometry_length
        #if len(ref_geometry) != len(strokes):
        #    print("skip:", f_name)
        #    continue
        # Test through row/col
        base_matrix = strokeErrorMatrix(strokes, ref_geometry, p_strokes, ref_progress_percentage)
        #print(base_matrix)
        error_maps = np.copy(base_matrix)
        row_stroke_map = np.full(len(strokes), -1)
        col_stroke_map = np.full(len(strokes), -1)
        row_mins = np.min(error_maps, axis=1)
        col_mins = np.min(error_maps, axis=0)
        compare_scores = []
        stroke_maps = {}
        # Iterate over every smallest error per row
        for row_min in range(len(ref_geometry)):
            coords = np.argwhere(error_maps == row_mins[row_min])
            if len(coords) > 1: # In cases where there are identical error values
                for coord in coords:
                    if not np.any(row_stroke_map == coord[0]):#row_stroke_map[coord[0]-1] != -1:
                        loc = coord
                        break
            else:
                loc = coords[0] # Find [row, col] index of current smallest error
            while row_stroke_map[loc[1]] != -1: # Make sure there's no overlap
                error_maps[loc[0]][loc[1]] = 10000
                loc[1] = np.argmin(error_maps[loc[0]])
            row_stroke_map[loc[1]] = loc[0]
            #print(row_stroke_map)
        if np.array2string(row_stroke_map) not in stroke_maps:
            stroke_maps[np.array2string(row_stroke_map)] = row_stroke_map
        error_maps = np.copy(base_matrix)
        # Iterate by column instead now
        for col_min in range(len(ref_geometry)):
            coords = np.argwhere(error_maps == col_mins[col_min])
            if len(coords) > 1: # In cases where there are identical error values
                for coord in coords:
                    if col_stroke_map[coord[1]-1] != -1:
                        loc = coord
                        break
            else:
                loc = coords[0] # Find [row, col] index of current smallest error
            while np.any(col_stroke_map == loc[0]): # Make sure there's no overlap
                error_maps[loc[0]][loc[1]] = 10000
                loc[0] = np.argmin(error_maps[:, loc[1]])
            col_stroke_map[loc[1]] = loc[0]
        if np.array2string(col_stroke_map) not in stroke_maps:
            stroke_maps[np.array2string(col_stroke_map)] = col_stroke_map
        j = False
        for s in stroke_maps.values():
            #print(s)
            heuristic_alignment = np.delete(s, np.where(s == -1))+1
            heuristic_xml = minXml(ref_char, bases, stroke_set, heuristic_alignment)
            try:
                heuristic_score = getXmlScore(heuristic_xml)
            except:
                j = True
                print(s)
                print("err:", f_name)
            compare_scores.append(heuristic_score)
        if j:
            print(compare_scores)
        # Greedy algorithm
        greedy_alignment = np.array(alignStrokes(strokes, ref_geometry, p_strokes, ref_progress_percentage))+1
        greedy_xml = minXml(ref_char, bases, stroke_set, greedy_alignment)
        greedy_score = getXmlScore(greedy_xml)
        compare_scores.append(greedy_score)
        #heuristic_scores.append(max(compare_scores))
        heuristic_scores[f_name] = max(compare_scores)
    return heuristic_scores

"""
First heuristic algorithm only iterating over columns
"""
def heuristic_col():
    ref_geometry, ref_progress_percentage, output_size = ref_data
    g_data, _, base_data, stroke_sets, _, f_names = char_data
    heuristic_scores = {}#[]
    for (geometry_length, bases, stroke_set, f_name) in zip(g_data, base_data, stroke_sets, f_names): # Iterating over gene data
        #print(f_name)
        strokes, p_strokes = geometry_length
        error_maps = strokeErrorMatrix(strokes, ref_geometry, p_strokes, ref_progress_percentage)
        #if len(ref_geometry) != len(strokes):
        #    print("skip", f_name)
        #    continue
        # Test through row/col
        col_stroke_map = np.full(len(strokes), -1)
        col_mins = np.min(error_maps, axis=0)
        compare_scores = []
        for col_min in range(len(ref_geometry)):
            coords = np.argwhere(error_maps == col_mins[col_min])
            if len(coords) > 1: # In cases where there are identical error values
                for coord in coords:
                    if col_stroke_map[coord[1]-1] != -1:
                        loc = coord
                        break
            else:
                loc = coords[0] # Find [row, col] index of current smallest error
            while np.any(col_stroke_map == loc[0]): # Make sure there's no overlap
                error_maps[loc[0]][loc[1]] = 10000
                loc[0] = np.argmin(error_maps[:, loc[1]])
            col_stroke_map[loc[1]] = loc[0]
        heuristic_alignment = np.delete(col_stroke_map, np.where(col_stroke_map == -1))+1
        heuristic_xml = minXml(ref_char, bases, stroke_set, heuristic_alignment)
        heuristic_score = getXmlScore(heuristic_xml)
        heuristic_scores[f_name] = heuristic_score
    return heuristic_scores

"""
Second heuristic algorithm: O(n!)

Methodology: Sum the error of every possible row permutation and choose the stroke order with the smallest total error
"""
def second_heuristic():
    ref_geometry, ref_progress_percentage, output_size = ref_data
    g_data, _, base_data, stroke_sets, _, f_names = char_data
    heuristic_scores = {}
    for (geometry_length, bases, stroke_set, f_name) in zip(g_data, base_data, stroke_sets, f_names):
        strokes, p_strokes = geometry_length
        error_maps = strokeErrorMatrix(strokes, ref_geometry, p_strokes, ref_progress_percentage)
        #np.fromiter(, dtype=tuple)
        least = 10000
        stroke_map = ()
        
        # n = len(ref_geometry)
        # perms = np.empty((np.math.factorial(n), n), dtype=np.uint8, order='F')
        # perms[0, 0] = 0
    
        # rows_to_copy = 1
        # for i in range(1, n):
        #     perms[:rows_to_copy, i] = i
        #     for j in range(1, i + 1):
        #         start_row = rows_to_copy * j
        #         end_row = rows_to_copy * (j + 1)
        #         splitter = i - j
        #         perms[start_row: end_row, splitter] = i
        #         perms[start_row: end_row, :splitter] = perms[:rows_to_copy, :splitter]  # left side
        #         perms[start_row: end_row, splitter + 1:i + 1] = perms[:rows_to_copy, splitter:i]  # right side
    
        #     rows_to_copy *= i + 1
        for priority in permutations(range(0, len(ref_geometry))): # Loop over every possible stroke priority
            #print(np.take(error_maps, priority, axis=1))
            s = np.sum(error_maps[np.arange(len(error_maps)), priority]) # Find the sum of the errors in this stroke priority
            if s < least: # Set stroke map if it has the smallest total error
                least = s
                stroke_map = priority
        heuristic_xml = minXml(ref_char, bases, stroke_set, np.argsort(stroke_map)+1)
        heuristic_score = getXmlScore(heuristic_xml)
        heuristic_scores[f_name] = heuristic_score
    return heuristic_scores

"""
Third heuristic algorithm: O(n!)

Methodology: Get the number of smallest errors present for every possible row permutation and choose the stroke order with the highest number of smallest errors
"""
def third_heuristic():
    ref_geometry, ref_progress_percentage, output_size = ref_data
    g_data, _, base_data, stroke_sets, _, f_names = char_data
    heuristic_scores = {}
    for (geometry_length, bases, stroke_set, f_name) in zip(g_data, base_data, stroke_sets, f_names):
        strokes, p_strokes = geometry_length
        error_maps = strokeErrorMatrix(strokes, ref_geometry, p_strokes, ref_progress_percentage)
        compare_scores = []
        stroke_maps = []#np.empty(20)
        smallerrs = np.min(error_maps, axis=1) # Get the smallest error for every row
        smallerr_count = 0
        for priority in permutations(range(0, len(ref_geometry))): # Loop over every possible stroke priority
            c = np.count_nonzero(smallerrs == error_maps[np.arange(len(error_maps)), priority]) # Get the number of elements from smallerrs present in this stroke priority
            if c > smallerr_count: # In the event that this priority has a larger number of smallerrs than the current lowest it should restart the test sample beginning with itself
                smallerr_count = c
                stroke_maps.clear()
                stroke_maps.append(np.argsort(priority))
            elif c == smallerr_count: # In the event this priority has the same number of smallerrs as the current lowest it should be added to the test sample
                stroke_maps.append(np.argsort(priority))
        for m in stroke_maps:
            heuristic_xml = minXml(ref_char, bases, stroke_set, m+1)
            heuristic_score = getXmlScore(heuristic_xml)
            compare_scores.append(heuristic_score)
        heuristic_scores[f_name] = max(compare_scores)
    return heuristic_scores

"""
Second and third heuristic algorithm combined
"""
def heuristic_comb():
    ref_geometry, ref_progress_percentage, output_size = ref_data
    g_data, _, base_data, stroke_sets, _, f_names = char_data
    heuristic_scores = {}
    for (geometry_length, bases, stroke_set, f_name) in zip(g_data, base_data, stroke_sets, f_names):
        strokes, p_strokes = geometry_length
        error_maps = strokeErrorMatrix(strokes, ref_geometry, p_strokes, ref_progress_percentage)
        compare_scores = []
        stroke_maps = []#np.empty(20)
        smallerrs = np.min(error_maps, axis=1)
        smallerr_count = 0
        least=10000
        stroke_map=()
        for priority in permutations(range(0, len(ref_geometry))): # Combine third_heuristic and second_heuristic
            check=error_maps[np.arange(len(error_maps)), priority]
            c = np.count_nonzero(smallerrs == check)#error_maps[np.arange(len(error_maps)), priority])
            if c > smallerr_count:
                smallerr_count = c
                stroke_maps.clear()
                stroke_maps.append(np.argsort(priority))
            elif c == smallerr_count:
                stroke_maps.append(np.argsort(priority))
            s = np.sum(check)
            if s < least:
                least = s
                stroke_map = priority
        stroke_maps.append(np.argsort(stroke_map))
        for m in stroke_maps:
            heuristic_xml = minXml(ref_char, bases, stroke_set, m+1)
            heuristic_score = getXmlScore(heuristic_xml)
            compare_scores.append(heuristic_score)
        heuristic_scores[f_name] = max(compare_scores)
    return heuristic_scores

"""
Fourth heuristic algorithm (not optimized but currently has the worst time complexity)

Methodology: Populate multiple stroke maps with the smallest errors without conflict as constants (similar to divide and conquer)
"""
def fourth_heuristic():
    ref_geometry, ref_progress_percentage, output_size = ref_data
    g_data, _, base_data, stroke_sets, _, f_names = char_data
    heuristic_scores = {}
    for (geometry_length, bases, stroke_set, f_name) in zip(g_data, base_data, stroke_sets, f_names):
        strokes, p_strokes = geometry_length
        error_maps = strokeErrorMatrix(strokes, ref_geometry, p_strokes, ref_progress_percentage)
        smallerrs = np.empty(len(ref_geometry), dtype=int)
        mins = np.min(error_maps, axis=1)
        for (i, v) in enumerate(mins):
            smallerrs[i] = np.argwhere(error_maps == v)[0][1] 
        conflicts = {} # a dict of conflicting smallest error indexes. the indexes of the potential stroke mapping are the keys and the potential values are the array value
        stroke_maps = []
        base_stroke_map = np.full(len(ref_geometry), -1, dtype=int)#np.empty(len(ref), dtype=int)#
        for num in range(len(ref_geometry)):
            locs = np.argwhere(smallerrs == num).flatten()
            if len(locs) == 1: # if there isn't any conflict with another stroke error make this index constant
                base_stroke_map[num] = locs[0]
                continue
            elif len(locs) == 0: # if the stroke mapping index isn't found it will be filled in last
                continue
            conflicts[num] = locs
        perm_list = []
        for p in conflicts.values():
            perm_list.append(permutations(p))
        for perm in product(*perm_list):
            s_map = np.copy(base_stroke_map)
            for (i, c) in enumerate(conflicts.keys()): # set the potential index
                s_map[c] = perm[i][0]
            missing_nums = []
            for num in range(len(ref_geometry)): # find the missing stroke map vals
                if not num in s_map:
                    missing_nums.append(num)
            for missing_perm in permutations(missing_nums): # fill in the missing vals
                final_map = np.copy(s_map)
                final_map[final_map == -1] = missing_perm
                stroke_maps.append(final_map)
        compare_scores = []
        for m in stroke_maps:
            heuristic_xml = minXml(ref_char, bases, stroke_set, m+1)
            heuristic_score = getXmlScore(heuristic_xml)
            compare_scores.append(heuristic_score)
        heuristic_scores[f_name] = max(compare_scores)
    return heuristic_scores

"""
Optimizing the third heuristic algorithm
"""
def third_heuristic2():
    ref_geometry, ref_progress_percentage, output_size = ref_data
    g_data, _, base_data, stroke_sets, _, f_names = char_data
    heuristic_scores = {}
    for (geometry_length, bases, stroke_set, f_name) in zip(g_data, base_data, stroke_sets, f_names):
        strokes, p_strokes = geometry_length
        error_maps = strokeErrorMatrix(strokes, ref_geometry, p_strokes, ref_progress_percentage)
        compare_scores = []
        stroke_maps = []#np.empty(20)
        np.bincount()
        smallerrs = np.min(error_maps, axis=1) # Get the smallest error for every row
        smallerr_count = len(ref)-np.sum(counts2[counts2>1]-1)
        for priority in permutations(range(0, len(ref_geometry))): # Loop over every possible stroke priority
            c = np.count_nonzero(smallerrs == error_maps[np.arange(len(error_maps)), priority]) # Get the number of elements from smallerrs present in this stroke priority
            if c > smallerr_count:
                smallerr_count = c
                stroke_maps.clear()
                stroke_maps.append(np.argsort(priority))
            elif c == smallerr_count: # In the event this priority has the same number of smallerrs as the current lowest it should be added to the test sample
                stroke_maps.append(np.argsort(priority))
        for m in stroke_maps:
            heuristic_xml = minXml(ref_char, bases, stroke_set, m+1)
            heuristic_score = getXmlScore(heuristic_xml)
            compare_scores.append(heuristic_score)
        heuristic_scores[f_name] = max(compare_scores)
    return heuristic_scores

"""
Exhausting every viable possibility of the stroke matrix. In theory this is the best that the stroke error functions can do in terms of accuracy.
There's still some way to go as this does not measure up to the exhaustive search (based on testing six stroke genes with various six stroke 
archetypes), meaning that a potential focus for future work could be tuning the stroke error functions.
"""
def dyn():
    ref, p_ref, _ = ref_data
    g_data, _, base_data, stroke_sets, _, f_names = char_data
    heuristic_scores = {}
    for (geometry_length, bases, stroke_set, f_name) in zip(g_data, base_data, stroke_sets, f_names):
        stroke_priority = permutations(range(0, len(ref)))
        compare_scores = []
        strokes, p_strokes = geometry_length
        b = 0
        stroke_maps = np.empty((0, len(ref)), int)
        #j = 0
        # Find candidate stroke orders
        #fromiter() = [(), (), (), etc.] then get indexes at values for each priority
        for priority in stroke_priority:
            error_maps = strokeErrorMatrix(strokes, ref, p_strokes, p_ref)
            stroke_map = np.full(len(strokes), -1)
            for i in priority:
                smallerror = np.argmin(error_maps[i]) # retrieve index of smallest error for current archetype stroke
                while(stroke_map[smallerror]!=-1):
                    # change small error so that we do not repeat over indexes that are already taken
                    # just keeps repeating until we land on an index that doesn't already have a value in its place
                    error_maps[i][smallerror] = 10000
                    smallerror = np.argmin(error_maps[i])
                stroke_map[smallerror] = i
            stroke_map = stroke_map[stroke_map!=-1]
            if not np.any(np.all(stroke_map == stroke_maps, axis=1)):
                #stroke_maps[j] = stroke_map
                stroke_maps = np.append(stroke_maps, np.array([stroke_map]), axis=0)
                #j += 1
        # Retrieve scores for each candidate stroke order
        for s in stroke_maps:
            heuristic_xml = minXml(ref_char, bases, stroke_set, s+1)
            heuristic_score = getXmlScore(heuristic_xml)
            compare_scores.append(heuristic_score)
        heuristic_scores[f_name] = max(compare_scores)
    return heuristic_scores

###

def run_benchmarks(funcs, trials):
    benchmarks = []
    for f in funcs:
        print(f"Running {f.__name__} benchmarks...")
        benchmarks.append(timeit.Timer(f, globals=locals()).timeit(number=int(trials)))
    return benchmarks

def format_benchmarks(funcs, benchmarks, wins, total, trials):
    print("")
    names = []
    for f in funcs:
        names.append("test_"+f.__name__)
    count = len(max(names, key=len))
    print("===SPEED BENCHMARKS===")
    for (n, b) in zip(names, benchmarks):
        print("{:>{}}:\t{} s".format(n, count, b[0]/trials))
    print("")
    print("===ACCURACY RESULTS===") # accuracy is measured by whether the algorithm's score for a gene matches the highest score computed among all algorithms being tested
    for (f, w) in zip(funcs, wins):
        print(f"{f.__name__} scored {w} out of {total} genes accurately.")

ref_dir = f'{str(Path.home())}/Stylus_Scoring_Generalization/Reference' # archetype directory
data_dir = f'{str(Path.home())}/Stylus_Scoring_Generalization/NewGenes' # gene directory

timeit.template = """
def inner(_it, _timer{init}):
    {setup}
    _t0 = _timer()
    for _i in _it:
        scores = {stmt}
    _t1 = _timer()
    return _t1-_t0, scores
"""

# currently non-working ref/gene pairings:
# completely broken: 5B57
# 4EFB/5408.2.15.gene, 6709/5408.2.3.gene, 6210/5408.2.15.gene, 5411/5408.2.3.gene
while True:
    print("Ctrl+C to exit")
    ref_char = input("Enter a reference character (example: 4EFB): ")
    print("Loading data...")
    ref_data = loadRef(ref_char, ref_dir)
    char_data = loadGeometryBases(data_dir, ref_data[2])
    trials = input("Enter number of trials: ")
    exhaustive_wins = 0
    greedy_wins = 0
    first_heuristic_wins = 0
    #heuristic_fallback_wins = 0
    #heuristic_col_wins = 0
    second_heuristic_wins = 0
    third_heuristic_wins = 0
    #heuristic_comb_wins = 0
    fourth_heuristic_wins = 0
    #dyn_wins = 0
    total = 0
    excl_exhaustive = input("Exclude exhaustive (T/F)? ")
    to_run = [exhaustive, greedy, first_heuristic, second_heuristic, third_heuristic, fourth_heuristic]#heuristic_fallback, heuristic_col, heuristic_comb, 
    if excl_exhaustive.lower() == "t":
        to_run.remove(exhaustive)
    benchmarks = run_benchmarks(to_run, trials)
    exhaustive_scores = {}
    if excl_exhaustive.lower() == "f":
        exhaustive_scores = benchmarks[to_run.index(exhaustive)][1]
    greedy_scores = benchmarks[to_run.index(greedy)][1]
    first_heuristic_scores = benchmarks[to_run.index(first_heuristic)][1]
    #heuristic_fallback_scores = benchmarks[to_run.index(heuristic_fallback)][1]
    #heuristic_col_scores = benchmarks[to_run.index(heuristic_col)][1]
    second_heuristic_scores = benchmarks[to_run.index(second_heuristic)][1]
    third_heuristic_scores = benchmarks[to_run.index(third_heuristic)][1]
    #heuristic_comb_scores = benchmarks[to_run.index(heuristic_comb)][1]
    fourth_heuristic_scores = benchmarks[to_run.index(fourth_heuristic)][1]
    #dyn_scores = benchmarks[to_run.index(dyn)][1]
    for f_name in char_data[5]:
        f_score = [greedy_scores[f_name], first_heuristic_scores[f_name], second_heuristic_scores[f_name], third_heuristic_scores[f_name], fourth_heuristic_scores[f_name]]#heuristic_fallback_scores[f_name], heuristic_col_scores[f_name], , heuristic_comb_scores[f_name]
        if excl_exhaustive.lower() == "f":
            f_score.insert(0, exhaustive_scores[f_name])
        best_score = max(*f_score)
        if excl_exhaustive.lower() == "f":
            if best_score == exhaustive_scores[f_name]:
                exhaustive_wins += 1
        if best_score == greedy_scores[f_name]:
            greedy_wins += 1
        if best_score == first_heuristic_scores[f_name]:
            first_heuristic_wins += 1
        # if best_score == heuristic_fallback_scores[f_name]:
        #     heuristic_fallback_wins += 1
        # if best_score == heuristic_col_scores[f_name]:
        #     heuristic_col_wins += 1
        if best_score == second_heuristic_scores[f_name]:
            second_heuristic_wins += 1
        if best_score == third_heuristic_scores[f_name]:
            third_heuristic_wins += 1
        # if best_score == heuristic_comb_scores[f_name]:
        #     heuristic_comb_wins += 1
        if best_score == fourth_heuristic_scores[f_name]:
            fourth_heuristic_wins += 1
        # if best_score == dyn_scores[f_name]:
        #     dyn_wins += 1
        f_len = [len(greedy_scores), len(first_heuristic_scores), len(second_heuristic_scores), len(third_heuristic_scores), len(fourth_heuristic_scores)]#len(heuristic_fallback_scores), len(heuristic_col_scores),  len(heuristic_comb_scores),
        if excl_exhaustive.lower() == "f":
            f_len.insert(0, len(exhaustive_scores))
        check = min(*f_len)
        if check == len(exhaustive_scores):
            total = len(exhaustive_scores)
        elif check == len(greedy_scores):
            total = len(greedy_scores)
        elif check == len(first_heuristic_scores):
            total = len(first_heuristic_scores)
        # elif check == len(heuristic_fallback_scores):
        #     total = len(heuristic_fallback_scores)
        # elif check == len(heuristic_col_scores):
        #     total = len(heuristic_col_scores)
        elif check == len(second_heuristic_scores):
            total = len(second_heuristic_scores)
        elif check == len(third_heuristic_scores):
            total = len(third_heuristic_scores)
        # elif check == len(heuristic_comb_scores):
        #     total = len(heuristic_comb_scores)
        elif check == len(fourth_heuristic_scores):
            total = len(fourth_heuristic_scores)
        # elif check == len(dyn_scores):
        #     total = len(dyn_scores)
    f_wins = [greedy_wins, first_heuristic_wins, second_heuristic_wins, third_heuristic_wins, fourth_heuristic_wins]#heuristic_fallback_wins, heuristic_col_wins, heuristic_comb_wins, 
    if excl_exhaustive.lower() == "f":
        f_wins.insert(0, exhaustive_wins)
    format_benchmarks(to_run, benchmarks, f_wins, total, int(trials))
    print("")
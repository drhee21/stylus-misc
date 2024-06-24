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
    
    # Edited from exhaustive.py
    def computeExhaustive(ref_char, f_read, data_dir, exhaust_dir = "Exhaustive", prog_interval = 100, save = True, xml_dir = "GenXml/Exhaustive", save_file = ""):
        n_strokes = len(ref_geometry)
        for i in range(len(g_data)):
            #print(f"Generating exhaustive scores for sample {f_read[i]}")
            bases = base_data[i]
            stroke_set = stroke_sets[i]
            exhaustive_alignments = permutations(range(1, n_strokes+1))
            exhaustive_scores = np.zeros(factorial(n_strokes))
            for j, p in enumerate(exhaustive_alignments):
                p_xml = minXml(ref_char, bases, stroke_set, p)
                exhaustive_scores[j] = getXmlScore(p_xml)
                #exhaustive_scores[j] = getXmlScore(p_xml, False, False)
                #if j%prog_interval == 0:
                #    print(f"Scoring permutation {j} of {len(exhaustive_scores)}")
            yield exhaustive_scores
            
    exhaustive_scores = {}
    for (gl, han_char, bases, f_name) in zip(g_data, han_chars, base_data, f_names):
        g, l = gl
        exhaust_maxes = []
        for e in computeExhaustive(ref_char, [f_name], data_dir, save = False, xml_dir = f'{str(Path.home())}/Stylus_Scoring_Generalization/GenXml/Exhaustive'):
            exhaust_maxes.append(e.max())
        original_score = np.max(exhaust_maxes)
        exhaustive_scores[f_name] = original_score
        #exhaustive_scores.append(original_score)
    return exhaustive_scores
    
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
def heuristic():
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
def heuristic_total():
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
def heuristic_small():
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
        for priority in permutations(range(0, len(ref_geometry))): # Combine heuristic_small and heuristic_total
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

Methodology: Populate multiple stroke maps with the smallest errors without conflict as constants
"""
def heuristic_e():
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
    heuristic_wins = 0
    heuristic_fallback_wins = 0
    heuristic_col_wins = 0
    heuristic_total_wins = 0
    heuristic_small_wins = 0
    heuristic_comb_wins = 0
    heuristic_e_wins = 0
    total = 0
    excl_exhaustive = input("Exclude exhaustive (T/F)? ")
    to_run = [exhaustive, greedy, heuristic, heuristic_fallback, heuristic_col, heuristic_total, heuristic_small, heuristic_comb, heuristic_e]
    if excl_exhaustive.lower() == "t":
        to_run.remove(exhaustive)
    benchmarks = run_benchmarks(to_run, trials)
    exhaustive_scores = {}
    if excl_exhaustive.lower() == "f":
        exhaustive_scores = benchmarks[0][1]
    greedy_scores = benchmarks[to_run.index(greedy)][1]
    heuristic_scores = benchmarks[to_run.index(heuristic)][1]
    heuristic_fallback_scores = benchmarks[to_run.index(heuristic_fallback)][1]
    heuristic_col_scores = benchmarks[to_run.index(heuristic_col)][1]
    heuristic_total_scores = benchmarks[to_run.index(heuristic_total)][1]
    heuristic_small_scores = benchmarks[to_run.index(heuristic_small)][1]
    heuristic_comb_scores = benchmarks[to_run.index(heuristic_comb)][1]
    heuristic_e_scores = benchmarks[to_run.index(heuristic_e)][1]
    for f_name in char_data[5]:
        f_score = [greedy_scores[f_name], heuristic_scores[f_name], heuristic_fallback_scores[f_name], heuristic_col_scores[f_name], heuristic_total_scores[f_name], heuristic_small_scores[f_name], heuristic_comb_scores[f_name], heuristic_e_scores[f_name]]
        if excl_exhaustive.lower() == "f":
            f_score.insert(0, exhaustive_scores)
        best_score = max(*f_score)
        if excl_exhaustive.lower() == "f":
            if best_score == exhaustive_scores[f_name]:
                exhaustive_wins += 1
        if best_score == greedy_scores[f_name]:
            greedy_wins += 1
        if best_score == heuristic_scores[f_name]:
            heuristic_wins += 1
        if best_score == heuristic_fallback_scores[f_name]:
            heuristic_fallback_wins += 1
        if best_score == heuristic_col_scores[f_name]:
            heuristic_col_wins += 1
        if best_score == heuristic_total_scores[f_name]:
            heuristic_total_wins += 1
        if best_score == heuristic_small_scores[f_name]:
            heuristic_small_wins += 1
        if best_score == heuristic_comb_scores[f_name]:
            heuristic_comb_wins += 1
        if best_score == heuristic_e_scores[f_name]:
            heuristic_e_wins += 1
        f_len = [len(greedy_scores), len(heuristic_scores), len(heuristic_fallback_scores), len(heuristic_col_scores), len(heuristic_total_scores), len(heuristic_small_scores), len(heuristic_comb_scores), len(heuristic_e_scores)]
        if excl_exhaustive.lower() == "f":
            f_len.insert(0, exhaustive_scores)
        check = min(*f_len)
        if check == len(exhaustive_scores):
            total == len(exhaustive_scores)
        elif check == len(greedy_scores):
            total = len(greedy_scores)
        elif check == len(heuristic_scores):
            total = len(heuristic_scores)
        elif check == len(heuristic_fallback_scores):
            total = len(heuristic_fallback_scores)
        elif check == len(heuristic_col_scores):
            total = len(heuristic_col_scores)
        elif check == len(heuristic_total_scores):
            total = len(heuristic_total_scores)
        elif check == len(heuristic_small_scores):
            total = len(heuristic_small_scores)
        elif check == len(heuristic_comb_scores):
            total = len(heuristic_comb_scores)
        elif check == len(heuristic_e_scores):
            total = len(heuristic_e_scores)
    f_wins = [greedy_wins, heuristic_wins, heuristic_fallback_wins, heuristic_col_wins, heuristic_total_wins, heuristic_small_wins, heuristic_comb_wins, heuristic_e_wins]
    if excl_exhaustive.lower() == "f":
        f_wins.insert(0, exhaustive_wins)
    format_benchmarks(to_run, benchmarks, f_wins, total, int(trials))
    print("")
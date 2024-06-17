from itertools import permutations
from math import factorial
from pathlib import Path

import os
import timeit

import numpy as np

from score_strokes import alignStrokes, greedyAlign2, strokeErrorMatrix
from xmlparse import extractBases, loadGeometryBases, loadRef, getXmlScore, minXml

## I can't stop Jupyter Notebook from printing out the genome fitness for every single trial so it's better to put the code in a Python script.
## Simple benchmark test for gene scoring algorithms.

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
    
def heuristic():
    ref_geometry, ref_progress_percentage, output_size = ref_data
    g_data, _, base_data, stroke_sets, _, f_names = char_data
    heuristic_scores = {}#[]
    for (geometry_length, bases, stroke_set, f_name) in zip(g_data, base_data, stroke_sets, f_names):
        #print(f_name)
        strokes, p_strokes = geometry_length
        base_matrix = strokeErrorMatrix(strokes, ref_geometry, p_strokes, ref_progress_percentage)
        if len(ref_geometry) != len(strokes):
            print("skip", f_name)
            continue
        # Test through row/col
        #print(base_matrix)
        error_maps = np.copy(base_matrix)
        row_stroke_map = np.full(len(strokes), -1)
        col_stroke_map = np.full(len(strokes), -1)
        row_mins = np.min(error_maps, axis=1)
        col_mins = np.min(error_maps, axis=0)
        compare_scores = []
        stroke_maps = {}
        # Iterate over every smallest error per row
        for row_min in row_mins:
            coords = np.argwhere(error_maps == row_min)
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
                # # remind program to switch the priority and repeat
            row_stroke_map[loc[1]] = loc[0]
            #print(row_stroke_map)
        if np.array2string(row_stroke_map) not in stroke_maps:
            stroke_maps[np.array2string(row_stroke_map)] = row_stroke_map
        # example: row 0's smallest error is at index 2 and so stroke_map[2] = 0
        # but row 4's smallest error is also at index 2
        # take row 0, recalculate the smallest error excluding index 2,
        # but it's too difficult so just permutation of all overlaps and rearrange them
        error_maps = np.copy(base_matrix)
        for col_min in col_mins:
            coords = np.argwhere(error_maps == col_min)
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
            heuristic_xml = minXml(ref_char, bases, stroke_set, s+1)
            try:
                heuristic_score = getXmlScore(heuristic_xml)
            except:
                print("err:", f_name)
            compare_scores.append(heuristic_score)
        #heuristic_scores.append(max(compare_scores))
        heuristic_scores[f_name] = max(compare_scores)
    return heuristic_scores

def heuristic_fallback():
    ref_geometry, ref_progress_percentage, output_size = ref_data
    g_data, _, base_data, stroke_sets, _, f_names = char_data
    heuristic_scores = {}#[]
    for (geometry_length, bases, stroke_set, f_name) in zip(g_data, base_data, stroke_sets, f_names):
        #print(f_name)
        strokes, p_strokes = geometry_length
        if len(ref_geometry) != len(strokes):
            print("skip:", f_name)
            continue
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
        for row_min in row_mins:
            coords = np.argwhere(error_maps == row_min)
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
                # # remind program to switch the priority and repeat
            row_stroke_map[loc[1]] = loc[0]
            #print(row_stroke_map)
        if np.array2string(row_stroke_map) not in stroke_maps:
            stroke_maps[np.array2string(row_stroke_map)] = row_stroke_map
        # example: row 0's smallest error is at index 2 and so stroke_map[2] = 0
        # but row 4's smallest error is also at index 2
        # take row 0, recalculate the smallest error excluding index 2,
        # but it's too difficult so just permutation of all overlaps and rearrange them
        error_maps = np.copy(base_matrix)
        for col_min in col_mins:
            coords = np.argwhere(error_maps == col_min)
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
            heuristic_xml = minXml(ref_char, bases, stroke_set, s+1)
            try:
                heuristic_score = getXmlScore(heuristic_xml)
            except:
                print("err:", f_name)
            compare_scores.append(heuristic_score)
        greedy_alignment = np.array(alignStrokes(strokes, ref_geometry, p_strokes, ref_progress_percentage))+1
        greedy_xml = minXml(ref_char, bases, stroke_set, greedy_alignment)
        greedy_score = getXmlScore(greedy_xml)
        compare_scores.append(greedy_score)
        #heuristic_scores.append(max(compare_scores))
        heuristic_scores[f_name] = max(compare_scores)
    return heuristic_scores

def run_benchmarks(funcs, trials):
    benchmarks = []
    for f in funcs:
        print(f"Running {f.__name__} benchmarks...")
        benchmarks.append(timeit.Timer(f, globals=locals()).timeit(number=int(trials)))
        print(f"Finished {f.__name__} benchmarks!")
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
    greedy_wins = 0
    heuristic_wins = 0
    heuristic_fallback_wins = 0
    excl_exhaustive = input("Exclude exhaustive (T/F)? ")
    if excl_exhaustive.lower() == "t":
        benchmarks = run_benchmarks([greedy, heuristic, heuristic_fallback], trials)
        greedy_scores = benchmarks[0][1]
        heuristic_scores = benchmarks[1][1]
        heuristic_fallback_scores = benchmarks[2][1]
        for f_name in char_data[5]:
            try:
                best_score = max(greedy_scores[f_name], heuristic_scores[f_name], heuristic_fallback_scores[f_name])
            except:
                continue
            if best_score == greedy_scores[f_name]:
                greedy_wins += 1
            if best_score == heuristic_scores[f_name]:
                heuristic_wins += 1
            if best_score == heuristic_fallback_scores[f_name]:
                heuristic_fallback_wins += 1
            check = min(len(greedy_scores), len(heuristic_scores), len(heuristic_fallback_scores))
            if check == len(greedy_scores):
                total = len(greedy_scores)
            elif check == len(heuristic_scores):
                total = len(heuristic_scores)
            elif check == len(heuristic_fallback_scores):
                total = len(heuristic_fallback_scores)
        format_benchmarks([greedy, heuristic, heuristic_fallback], benchmarks, [greedy_wins, heuristic_wins, heuristic_fallback_wins], total, int(trials))
    else:
        exhaustive_wins = 0
        benchmarks = run_benchmarks([exhaustive, greedy, heuristic, heuristic_fallback], trials)
        exhaustive_scores = benchmarks[0][1]
        greedy_scores = benchmarks[1][1]
        heuristic_scores = benchmarks[2][1]
        heuristic_fallback_scores = benchmarks[3][1]
        for f_name in char_data[5]:
            try:
                best_score = max(exhaustive_scores[f_name], greedy_scores[f_name], heuristic_scores[f_name], heuristic_fallback_scores[f_name])
            except:
                continue
            if best_score == exhaustive_scores[f_name]:
                exhaustive_wins += 1
            if best_score == greedy_scores[f_name]:
                greedy_wins += 1
            if best_score == heuristic_scores[f_name]:
                heuristic_wins += 1
            if best_score == heuristic_fallback_scores[f_name]:
                heuristic_fallback_wins += 1
            check = min(len(exhaustive_scores), len(greedy_scores), len(heuristic_scores), len(heuristic_fallback_scores))
            if check == len(exhaustive_scores):
                total = len
            elif check == len(greedy_scores):
                total = len(greedy_scores)
            elif check == len(heuristic_scores):
                total = len(heuristic_scores)
            elif check == len(heuristic_fallback_scores):
                total = len(heuristic_fallback_scores)
        format_benchmarks([exhaustive, greedy, heuristic, heuristic_fallback], benchmarks, [exhaustive_wins, greedy_wins, heuristic_wins, heuristic_fallback_wins], total, int(trials))
    print("")
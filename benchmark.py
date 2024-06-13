from itertools import permutations
from math import factorial
from pathlib import Path

import os
import timeit

import numpy as np

from score_strokes import alignStrokes, greedyAlign2, strokeErrorMatrix
from xmlparse import extractBases, loadGeometryBases, loadRef, getXmlScore, minXml

## I can't stop Jupyter Notebook from printing out the genome fitness for every single trial so it's better to put the code in a Python script.
## Simple benchmark test for gene scoring algorithms comparing six stroke genes.




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
            
    exhaustive_scores = []
    for (gl, han_char, bases, f_name) in zip(g_data, han_chars, base_data, f_names):
        g, l = gl
        exhaust_maxes = []
        for e in computeExhaustive(ref_char, [f_name], data_dir, save = False, xml_dir = f'{str(Path.home())}/Stylus_Scoring_Generalization/GenXml/Exhaustive'):
            exhaust_maxes.append(e.max())
        original_score = np.max(exhaust_maxes)
        exhaustive_scores.append(original_score)
    return exhaustive_scores

def greedy():
    ref_geometry, ref_progress_percentage, output_size = ref_data
    g_data, _, base_data, stroke_sets, _, f_names = char_data
    heuristic_scores = []
    for (geometry_length, bases, stroke_set, f_name) in zip(g_data, base_data, stroke_sets, f_names):
        geometry, progress_percentage = geometry_length
        heuristic_alignment = np.array(alignStrokes(geometry, ref_geometry, progress_percentage, ref_progress_percentage))+1
        heuristic_xml = minXml(ref_char, bases, stroke_set, heuristic_alignment)
        heuristic_score = getXmlScore(heuristic_xml)
        heuristic_scores.append(heuristic_score)
    return heuristic_scores
    
def heuristic():
    ref_geometry, ref_progress_percentage, output_size = ref_data
    g_data, _, base_data, stroke_sets, _, f_names = char_data
    heuristic_scores = []
    for (geometry_length, bases, stroke_set, f_name) in zip(g_data, base_data, stroke_sets, f_names):
        #print(f_name)
        strokes, p_strokes = geometry_length
        if len(ref_geometry) != len(strokes):
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
        heuristic_scores.append(max(compare_scores))
    return heuristic_scores

def heuristic_fallback():
    ref_geometry, ref_progress_percentage, output_size = ref_data
    g_data, _, base_data, stroke_sets, _, f_names = char_data
    heuristic_scores = []
    for (geometry_length, bases, stroke_set, f_name) in zip(g_data, base_data, stroke_sets, f_names):
        #print(f_name)
        strokes, p_strokes = geometry_length
        if len(ref_geometry) != len(strokes):
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
        heuristic_scores.append(max(compare_scores))
    return heuristic_scores

#def format_benchmarks():

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


while True:
    print("Ctrl+C to exit")
    ref_char = input("Enter a reference character (example: 4EFB): ")
    print("Loading data...")
    ref_data = loadRef(ref_char, ref_dir)
    char_data = loadGeometryBases(data_dir, ref_data[2])
    trials = input("Enter number of trials: ")
    print("Running exhaustive benchmarks...")
    exhaustive_benchmark = timeit.Timer(exhaustive, globals=locals()).timeit(number=int(trials))
    print("Finished exhaustive benchmarks!")
    print("Running greedy benchmarks...")
    greedy_benchmark = timeit.Timer(greedy, globals=locals()).timeit(number=int(trials))
    print("Finished greedy benchmarks!")
    print("Running heuristic benchmarks...")
    heuristic_benchmark = timeit.Timer(heuristic, globals=locals()).timeit(number=int(trials))
    print("Finished heuristic benchmarks!")
    print("Running heuristic_fallback benchmarks...")
    heuristic_fallback_benchmark = timeit.Timer(heuristic_fallback, globals=locals()).timeit(number=int(trials))
    print("Finished heuristic_fallback benchmarks!")
    #print("{:>20}:\t{:0.5f}".format(name, mean(timings)))
    exhaustive_wins = 0
    greedy_wins = 0
    heuristic_wins = 0
    heuristic_fallback_wins = 0
    for (exhaustive_score, greedy_score, heuristic_score, heuristic_fallback_score) in zip(exhaustive_benchmark[1], greedy_benchmark[1], heuristic_benchmark[1], heuristic_fallback_benchmark[1]):
        best_score = max(exhaustive_score, greedy_score, heuristic_score, heuristic_fallback_score)
        if best_score == exhaustive_score:
            exhaustive_wins += 1
        if best_score == greedy_score:
            greedy_wins += 1
        if best_score == heuristic_score:
            heuristic_wins += 1
        if best_score == heuristic_fallback_score:
            heuristic_fallback_wins += 1
    print(f"Exhaustive ({exhaustive_wins}/{len(char_data[0])}): {exhaustive_benchmark[0]}s")
    print(f"Greedy ({greedy_wins}/{len(char_data[0])}): {greedy_benchmark[0]}s")
    print(f"Heuristic ({heuristic_wins}/{len(char_data[0])}): {heuristic_benchmark[0]}s")
    print(f"Heuristic Fallback ({heuristic_fallback_wins}/{len(char_data[0])}): {heuristic_fallback_benchmark[0]}s")
    print("")
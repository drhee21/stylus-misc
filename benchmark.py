from itertools import permutations
from math import factorial
from time import time
from pathlib import Path

import os

import numpy as np

from score_strokes import alignStrokes, greedyAlign2, strokeErrorMatrix
from xmlparse import extractBases, loadGeometryBases, loadRef, getXmlScore, minXml

## I can't stop Jupyter Notebook from printing out the genome fitness for every single trial so it's better to put the code in a Python script.
## Simple benchmark test for gene scoring algorithms comparing six stroke genes.

# Edited from exhaustive.py
def computeExhaustive(ref_char, f_read, data_dir, exhaust_dir = "Exhaustive", prog_interval = 100, save = True, xml_dir = "GenXml/Exhaustive", save_file = ""):
    ref_g, ref_l, output_size = loadRef(ref_char, ref_dir)
    g_data, _, base_data, stroke_sets, _, f_names = loadGeometryBases(data_dir, output_size, f_read = f_read)
    n_strokes = len(ref_g)
    for i in range(len(g_data)):
        #print(f"Generating exhaustive scores for sample {f_read[i]}")
        bases = base_data[i]
        stroke_set = stroke_sets[i]
        exhaustive_alignments = permutations(range(1, n_strokes+1))
        exhaustive_scores = np.zeros(factorial(n_strokes))
        for j, p in enumerate(exhaustive_alignments):
            p_xml = minXml(ref_char, bases, stroke_set, p)
            exhaustive_scores[j] = getXmlScore(p_xml, f"{xml_dir}/{i}_{j}_{f_read[i]}", f"{xml_dir}/{i}_{j}_min_{f_read[i]}")
            #exhaustive_scores[j] = getXmlScore(p_xml, False, False)
            #if j%prog_interval == 0:
            #    print(f"Scoring permutation {j} of {len(exhaustive_scores)}")
        if save:
            if save_file == "":
                f_name_cleaned = f_read[i].replace("/", "_")
                f"{exhaust_dir}/exhaust_{ref_char}_{f_name_cleaned}.npy"
            print(f"Wrote exhaustive scores to {save_file}")
            np.save(save_file, exhaustive_scores)
        yield exhaustive_scores

"""
Function to compare two heuristic algorithms' accuracy and performance.
"""
def compareHeuristic(algo1, algo2, ref_data, char_data):
    ref_geometry, ref_progress_percentage, output_size = ref_data
    g_data, _, base_data, stroke_sets, _, f_names = char_data
    
    
    def benchmark(algo):
        heuristic_scores = []
        heuristic_alignments = []
        for (geometry_length, bases, stroke_set, f_name) in zip(g_data, base_data, stroke_sets, f_names):
            geometry, progress_percentage = geometry_length
            heuristic_alignment = np.array(algo(geometry, ref_geometry, progress_percentage, ref_progress_percentage))+1
            heuristic_alignments.append(heuristic_alignment)
            heuristic_xml = minXml(ref_char, bases, stroke_set, heuristic_alignment)
            heuristic_score = getXmlScore(heuristic_xml)
            heuristic_scores.append(heuristic_score)
        return heuristic_scores, heuristic_alignments

    wins1 = 0
    wins2 = 0
    print("Running first algorithm...")
    init1 = time()
    scores1, align1 = benchmark(algo1)
    results1 = time()-init1
    print("Running second algorithm...")
    init2 = time()
    scores2, align2 = benchmark(algo2)
    results2 = time()-init2

    for (score1, score2) in zip(scores1, scores2):
        if score1 > score2:
            wins1 += 1
        elif score2 > score1:
            wins2 += 1

    print("The first algorithm took", results1, "seconds to execute.")
    print("The second algorithm took", results2, "seconds to execute.")
    print("The first algorithm scored", wins1, "genes more accurately than the second algorithm.")
    print("The second algorithm scored", wins2, "genes more accurately than the first algorithm.")
    print("The first and second algorithm scored", len(scores1)-wins1-wins2, "genes identically.")

"""
Function to compare a heuristic algorithm's accuracy and performance against the exhaustive search.
"""
def compareExhaustive(algo, ref_data, char_data):
    ref_geometry, ref_progress_percentage, output_size = ref_data
    g_data, han_chars, base_data, stroke_sets, _, f_names = char_data
    
    
    def heuristic(algo):
        heuristic_scores = []
        heuristic_alignments = []
        for (geometry_length, bases, stroke_set, f_name) in zip(g_data, base_data, stroke_sets, f_names):
            geometry, progress_percentage = geometry_length
            heuristic_alignment = np.array(algo(geometry, ref_geometry, progress_percentage, ref_progress_percentage))+1
            heuristic_alignments.append(heuristic_alignment)
            heuristic_xml = minXml(ref_char, bases, stroke_set, heuristic_alignment)
            heuristic_score = getXmlScore(heuristic_xml)
            heuristic_scores.append(heuristic_score)
        return heuristic_scores, heuristic_alignments

    def exhaustive():
        exhaustive_scores = []
        for (gl, han_char, bases, f_name) in zip(g_data, han_chars, base_data, f_names):
            g, l = gl
            exhaust_maxes = []
            for e in computeExhaustive(ref_char, [f_name], data_dir, save = False, xml_dir = f'{str(Path.home())}/Stylus_Scoring_Generalization/GenXml/Exhaustive'):
                exhaust_maxes.append(e.max())
            original_score = np.max(exhaust_maxes)
            exhaustive_scores.append(original_score)
        return exhaustive_scores

    wins1 = 0
    wins2 = 0
    print("Running greedy algorithm...")
    init1 = time()
    scores1, _ = heuristic(algo)
    results1 = time()-init1
    print("Running exhaustive search...")
    init2 = time()
    scores2 = exhaustive()
    results2 = time()-init2

    for (score1, score2) in zip(scores1, scores2):
        if score1 > score2:
            wins1 += 1
        elif score2 > score1:
            wins2 += 1
            
    print("The greedy algorithm took", results1, "seconds to execute.")
    print("The exhaustive search took", results2, "seconds to execute.")
    print("The greedy algorithm scored", wins1, "genes more accurately than the exhaustive search.")
    print("The exhaustive search scored", wins2, "genes more accurately than the greedy algorithm.")
    print("The greedy algorithm and exhaustive search scored", len(scores1)-wins1-wins2, "genes identically.")


def compareHeuristic2(algo, ref_char, ref_data, char_data):
    ref_geometry, ref_progress_percentage, output_size = ref_data
    g_data, han_chars, base_data, stroke_sets, _, f_names = char_data
    
    
    def heuristic(algo):
        heuristic_scores = []
        heuristic_alignments = []
        for (geometry_length, bases, stroke_set, f_name) in zip(g_data, base_data, stroke_sets, f_names):
            geometry, progress_percentage = geometry_length
            heuristic_alignment = np.array(algo(geometry, ref_geometry, progress_percentage, ref_progress_percentage))+1
            heuristic_alignments.append(heuristic_alignment)
            heuristic_xml = minXml(ref_char, bases, stroke_set, heuristic_alignment)
            heuristic_score = getXmlScore(heuristic_xml)
            heuristic_scores.append(heuristic_score)
        return heuristic_scores, heuristic_alignments

    
    def newHeuristic():
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
    
    # def dynamic():
    #     heuristic_scores = []
    #     for (geometry_length, bases, stroke_set, f_name) in zip(g_data, base_data, stroke_sets, f_names):
    #         stroke_priority = permutations(range(0, len(ref_geometry)))
    #         stroke_maps = []
    #         compare_scores = []
    #         strokes, p_strokes = geometry_length
    #         # Find candidate stroke orders
    #         for priority in stroke_priority:
    #             error_maps = strokeErrorMatrix(strokes, ref_geometry, p_strokes, ref_progress_percentage)
    #             stroke_map = np.full(len(strokes), -1)
    #             for i in priority:
    #                 smallerror = np.argmin(error_maps[i]) # retrieve index of smallest error for current archetype stroke
    #                 while(stroke_map[smallerror]!=-1):
    #                     # change small error so that we do not repeat over indexes that are already taken
    #                     # just keeps repeating until we land on an index that doesn't already have a value in its place
    #                     error_maps[i][smallerror] = 10000
    #                     smallerror = np.argmin(error_maps[i])
    #                 stroke_map[smallerror] = i
    #             if not any(np.array_equal(stroke_map, m) for m in stroke_maps):
    #                 stroke_map = np.array([n for n in stroke_map if n != -1])
    #                 stroke_maps.append(stroke_map)
    #         # Retrieve scores for each candidate stroke order
    #         for s in stroke_maps:
    #             heuristic_xml = minXml(ref_char, bases, stroke_set, np.array(s)+1)
    #             heuristic_score = getXmlScore(heuristic_xml)
    #             compare_scores.append(heuristic_score)
    #         heuristic_scores.append(max(compare_scores))
    #     return heuristic_scores
    

    wins1 = 0
    wins2 = 0
    print("Running greedy algorithm...")
    init1 = time()
    scores1, _ = heuristic(algo)
    results1 = time()-init1
    print("Running heuristic algorithm...")
    init2 = time()
    scores2 = newHeuristic()#dynamic()
    results2 = time()-init2

    for (score1, score2) in zip(scores1, scores2):
        if score1 > score2:
            wins1 += 1
        elif score2 > score1:
            wins2 += 1

    print("The greedy algorithm took", results1, "seconds to execute.")
    print("The heuristic algorithm took", results2, "seconds to execute.")
    print("The greedy algorithm scored", wins1, "genes more accurately than the heuristic algorithm.")
    print("The heuristic algorithm scored", wins2, "genes more accurately than the greedy algorithm.")
    print("The greedy algorithm and heuristic algorithm scored", len(scores1)-wins1-wins2, "genes identically.")


def compareExhaustive2(ref_char, ref_data, char_data):
    ref_geometry, ref_progress_percentage, output_size = ref_data
    g_data, han_chars, base_data, stroke_sets, _, f_names = char_data
    
    
    def newHeuristic():
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
    
    # def dynamic():
    #     heuristic_scores = []
    #     for (geometry_length, bases, stroke_set, f_name) in zip(g_data, base_data, stroke_sets, f_names):
    #         stroke_priority = permutations(range(0, len(ref_geometry)))
    #         stroke_maps = []
    #         compare_scores = []
    #         strokes, p_strokes = geometry_length
    #         # Find candidate stroke orders
    #         for priority in stroke_priority:
    #             error_maps = strokeErrorMatrix(strokes, ref_geometry, p_strokes, ref_progress_percentage)
    #             stroke_map = np.full(len(strokes), -1)
    #             for i in priority:
    #                 smallerror = np.argmin(error_maps[i]) # retrieve index of smallest error for current archetype stroke
    #                 while(stroke_map[smallerror]!=-1):
    #                     # change small error so that we do not repeat over indexes that are already taken
    #                     # just keeps repeating until we land on an index that doesn't already have a value in its place
    #                     error_maps[i][smallerror] = 10000
    #                     smallerror = np.argmin(error_maps[i])
    #                 stroke_map[smallerror] = i
    #             if not any(np.array_equal(stroke_map, m) for m in stroke_maps):
    #                 stroke_map = np.array([n for n in stroke_map if n != -1])
    #                 stroke_maps.append(stroke_map)
    #         # Retrieve scores for each candidate stroke order
    #         for s in stroke_maps:
    #             heuristic_xml = minXml(ref_char, bases, stroke_set, np.array(s)+1)
    #             heuristic_score = getXmlScore(heuristic_xml)
    #             compare_scores.append(heuristic_score)
    #         heuristic_scores.append(max(compare_scores))
    #     return heuristic_scores

    def exhaustive():
        exhaustive_scores = []
        for (gl, han_char, bases, f_name) in zip(g_data, han_chars, base_data, f_names):
            g, l = gl
            exhaust_maxes = []
            for e in computeExhaustive(ref_char, [f_name], data_dir, save = False, xml_dir = f'{str(Path.home())}/Stylus_Scoring_Generalization/GenXml/Exhaustive'):
                exhaust_maxes.append(e.max())
            original_score = np.max(exhaust_maxes)
            exhaustive_scores.append(original_score)
        return exhaustive_scores

    wins1 = 0
    wins2 = 0
    print("Running heuristic algorithm...")
    init1 = time()
    scores1 = newHeuristic()#dynamic()
    results1 = time()-init1
    print("Running exhaustive search...")
    init2 = time()
    scores2 = exhaustive()
    results2 = time()-init2

    for (score1, score2) in zip(scores1, scores2):
        if score1 > score2:
            wins1 += 1
        elif score2 > score1:
            wins2 += 1

    print("The heuristic algorithm took", results1, "seconds to execute.")
    print("The exhaustive search took", results2, "seconds to execute.")
    print("The heuristic algorithm scored", wins1, "genes more accurately than the exhaustive search.")
    print("The exhaustive search scored", wins2, "genes more accurately than the heuristic algorithm.")
    print("The heuristic algorithm and exhaustive search scored", len(scores1)-wins1-wins2, "genes identically.")


ref_dir = f'{str(Path.home())}/Stylus_Scoring_Generalization/Reference' # archetype directory
data_dir = f'{str(Path.home())}/Stylus_Scoring_Generalization/NewGenes' # gene directory
ref_char = "5B89"



ref_data = loadRef(ref_char, ref_dir)
char_data = loadGeometryBases(data_dir, ref_data[2])

while True:
    print("1. Exhaustive vs. Greedy\n2. Exhaustive vs. Heuristic\n3. Greedy vs. Heuristic\n4. Heuristic vs. Heuristic")
    print("Ctrl+C to exit")
    ref_char = input("Enter a reference character (example: 4EFB): ")
    print("Loading data...")
    ref_data = loadRef(ref_char, ref_dir)
    char_data = loadGeometryBases(data_dir, ref_data[2])
    c = input("Choose an option: ")
    if c == "1":
        compareExhaustive(alignStrokes, ref_data, char_data)
    elif c == "2":
        compareExhaustive2(ref_char, ref_data, char_data)
    elif c == "3":
        compareHeuristic2(alignStrokes, ref_char, ref_data, char_data)
    elif c == "4":
        pass
        #compareHeuristic(alignStrokes, greedyAlign2, ref_data, char_data, int(trials))
    print("")
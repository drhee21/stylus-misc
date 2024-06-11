from itertools import permutations
from math import factorial
from pathlib import Path

import os
import time
import timeit

import numpy as np

from score_strokes import alignStrokes, greedyAlign2, strokeErrorMatrix
from xmlparse import extractBases, loadGeometryBases, loadRef, getXmlScore, minXml

## I can't stop Jupyter Notebook from printing out the genome fitness for every single trial so it's better to put the code in a Python script.

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
def compareHeuristic(algo1, algo2, ref_data, char_data, trials):
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
    scores1, align1 = benchmark(algo1)
    scores2, align2 = benchmark(algo2)

    for (score1, score2) in zip(scores1, scores2):
        if score1 > score2:
            wins1 += 1
        elif score2 > score1:
            wins2 += 1

    print("Running first algorithm...")
    results1 = timeit.timeit("benchmark(algo1)", number=trials, globals=locals())
    print("Running second algorithm...")
    results2 = timeit.timeit("benchmark(algo2)", number=trials, globals=locals())
    print("The first algorithm took", results1, "seconds to execute", trials, "times.")
    print("The second algorithm took", results2, "seconds to execute", trials, "times.")
    print("The first algorithm scored", wins1, "genes more accurately than the second algorithm.")
    print("The second algorithm scored", wins2, "genes more accurately than the first algorithm.")
    print("The first and second algorithm scored", len(scores1)-wins1-wins2, "genes identically.")

"""
Function to compare a heuristic algorithm's accuracy and performance against the exhaustive search.
"""
def compareExhaustive(algo, ref_data, char_data, trials):
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
    scores1, _ = heuristic(algo)
    scores2 = exhaustive()

    for (score1, score2) in zip(scores1, scores2):
        if score1 > score2:
            wins1 += 1
        elif score2 > score1:
            wins2 += 1

    print("Running greedy algorithm...")
    results1 = timeit.timeit("heuristic(algo)", number=trials, globals=locals())
    print("Running exhaustive search...")
    results2 = timeit.timeit("exhaustive()", number=trials, globals=locals())
    print("The greedy algorithm took", results1, "seconds to execute", trials, "times.")
    print("The exhaustive search took", results2, "seconds to execute", trials, "times.")
    print("The greedy algorithm scored", wins1, "genes more accurately than the exhaustive search.")
    print("The exhaustive search scored", wins2, "genes more accurately than the greedy algorithm.")
    print("The greedy algorithm and exhaustive search scored", len(scores1)-wins1-wins2, "genes identically.")


def compareDynamic(algo, ref_char, ref_data, char_data, trials):
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

    
    
    def dynamic(ref_char, ref_data, char_data):
        ref, p_ref, _ = ref_data
        g_data, _, base_data, stroke_sets, _, f_names = char_data
        heuristic_scores = []
        for (geometry_length, bases, stroke_set, f_name) in zip(g_data, base_data, stroke_sets, f_names):
            stroke_priority = permutations(range(0, len(ref)))
            stroke_maps = []
            compare_scores = []
            strokes, p_strokes = geometry_length
            # Find candidate stroke orders
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
                if not any(np.array_equal(stroke_map, m) for m in stroke_maps):
                    stroke_map = np.array([n for n in stroke_map if n != -1])
                    stroke_maps.append(stroke_map)
            # Retrieve scores for each candidate stroke order
            for s in stroke_maps:
                heuristic_xml = minXml(ref_char, bases, stroke_set, np.array(s)+1)
                heuristic_score = getXmlScore(heuristic_xml)
                compare_scores.append(heuristic_score)
            heuristic_scores.append(max(compare_scores))
        return heuristic_scores
    

    wins1 = 0
    wins2 = 0
    scores1, _ = heuristic(algo)
    scores2 = dynamic(ref_char, ref_data, char_data)

    for (score1, score2) in zip(scores1, scores2):
        if score1 > score2:
            wins1 += 1
        elif score2 > score1:
            wins2 += 1

    print("Running greedy algorithm...")
    results1 = timeit.timeit("heuristic(algo)", number=trials, globals=locals())
    print("Running dynamic algorithm...")
    results2 = timeit.timeit("dynamic(ref_char, ref_data, char_data)", number=trials, globals=locals())
    print("The greedy algorithm took", results1, "seconds to execute", trials, "times.")
    print("The dynamic algorithm took", results2, "seconds to execute", trials, "times.")
    print("The greedy algorithm scored", wins1, "genes more accurately than the dynamic algorithm.")
    print("The dynamic algorithm scored", wins2, "genes more accurately than the greedy algorithm.")
    print("The greedy algorithm and dynamic algorithm scored", len(scores1)-wins1-wins2, "genes identically.")


def compareDynamicExhaustive(ref_char, ref_data, char_data, trials):
    ref_geometry, ref_progress_percentage, output_size = ref_data
    g_data, han_chars, base_data, stroke_sets, _, f_names = char_data
    
    
    def dynamic(ref_char, ref_data, char_data):
        ref, p_ref, _ = ref_data
        g_data, _, base_data, stroke_sets, _, f_names = char_data
        heuristic_scores = []
        for (geometry_length, bases, stroke_set, f_name) in zip(g_data, base_data, stroke_sets, f_names):
            stroke_priority = permutations(range(0, len(ref)))
            stroke_maps = []
            compare_scores = []
            strokes, p_strokes = geometry_length
            # Find candidate stroke orders
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
                if not any(np.array_equal(stroke_map, m) for m in stroke_maps):
                    stroke_map = np.array([n for n in stroke_map if n != -1])
                    stroke_maps.append(stroke_map)
            # Retrieve scores for each candidate stroke order
            for s in stroke_maps:
                heuristic_xml = minXml(ref_char, bases, stroke_set, np.array(s)+1)
                heuristic_score = getXmlScore(heuristic_xml)
                compare_scores.append(heuristic_score)
            heuristic_scores.append(max(compare_scores))
        return heuristic_scores

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
    scores1 = dynamic(ref_char, ref_data, char_data)
    scores2 = exhaustive()

    for (score1, score2) in zip(scores1, scores2):
        if score1 > score2:
            wins1 += 1
        elif score2 > score1:
            wins2 += 1

    print("Running dynamic algorithm...")
    results1 = timeit.timeit("dynamic(ref_char, ref_data, char_data)", number=trials, globals=locals())
    print("Running exhaustive search...")
    results2 = timeit.timeit("exhaustive()", number=trials, globals=locals())
    print("The dynamic algorithm took", results1, "seconds to execute", trials, "times.")
    print("The exhaustive search took", results2, "seconds to execute", trials, "times.")
    print("The dynamic algorithm scored", wins1, "genes more accurately than the exhaustive search.")
    print("The exhaustive search scored", wins2, "genes more accurately than the dynamic algorithm.")
    print("The dynamic algorithm and exhaustive search scored", len(scores1)-wins1-wins2, "genes identically.")


ref_dir = f'{str(Path.home())}/Stylus_Scoring_Generalization/Reference' # archetype directory
data_dir = f'{str(Path.home())}/Stylus_Scoring_Generalization/NewGenes' # gene directory
ref_char = "6709"



ref_data = loadRef(ref_char, ref_dir)
char_data = loadGeometryBases(data_dir, ref_data[2])

while True:
    print("1. Exhaustive vs. Greedy\n2. Exhaustive vs. Dynamic\n3. Greedy vs. Dynamic\n4. Heuristic vs. Heuristic")
    print("Ctrl+C to exit")
    c = input("Choose an option: ")
    if c == "1":
        trials = input("Amount of trials: ")
        compareExhaustive(alignStrokes, ref_data, char_data, int(trials))
    elif c == "2":
        trials = input("Amount of trials: ")
        compareDynamicExhaustive(ref_char, ref_data, char_data, int(trials))
    elif c == "3":
        trials = input("Amount of trials: ")
        compareDynamic(alignStrokes, ref_char, ref_data, char_data, int(trials))
    elif c == "4":
        trials = input("Amount of trials: ")
        #compareHeuristic(alignStrokes, greedyAlign2, ref_data, char_data, int(trials))
    print("")
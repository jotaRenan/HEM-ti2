import math
import random
import time
import os
import itertools
from functools import reduce

RUNS = 1

def execute_runs(dir, att = False):
  tot = 0
  for filename in os.listdir(dir):
    if tot > 0:
      break
    tot += 1
    coords = []
    line_count = 0
    with open(dir+filename) as file:
      for line in file:
        line = line.strip()
        if line == 'EOF':
          break
        if line_count < 6:
          line_count += 1
        else:
          [_,  x, y] = line.split()
          x = float(x)
          y = float(y)
          coords.append((x, y)) 
    distances_matrix = generate_distances_matrix(coords, att)

    distances_sum = 0.0
    time_sum = 0.0
    for _ in range(RUNS):
      start = time.time()
      result, path = run_vnd_heuristic(distances_matrix)
      elapsed = time.time() - start
      distances_sum += result
      time_sum += elapsed

    print(filename, round(distances_sum/float(RUNS)), f'{((time_sum/float(RUNS))*float(RUNS) * 1000):.2f}', sep="\t")

def calc_distance(p1, p2, is_pseudo_euclidian):
  if is_pseudo_euclidian:
    xd = p1[0]-p2[0]
    yd = p1[1]-p2[1]
    r = math.sqrt((xd*xd + yd*yd)/10.0)
    t = int(round(r))
    if t < r:
      return t+1
    else:
      return t 
  else:
    xd = p1[0]-p2[0]
    yd = p1[1]-p2[1]
    return int(round(math.sqrt(xd*xd + yd*yd)))

def generate_distances_matrix(coordinates, is_pseudo_euclidian):
  distances_matrix = []
  numberOfEdges = len(coordinates)
  for i in range(numberOfEdges):
    distances_i = []
    for j in range(numberOfEdges):
      distance_ij = calc_distance(coordinates[i], coordinates[j], is_pseudo_euclidian)
      distances_i.append(distance_ij)
    distances_matrix.append(distances_i)
  return distances_matrix


def run_nn_heuristic(distances_matrix):
  starting_city = random.randint(0, len(distances_matrix) - 1)
  current_city = starting_city
  visited_cities_indexes = set()
  total_distance = 0
  solution = []

  for _ in range(len(distances_matrix)):
    visited_cities_indexes.add(current_city)
    solution.append(current_city)
    nearest_distance = math.inf

    for neighbor_index in range(len(distances_matrix)):
      distance_to_neighbor = distances_matrix[current_city][neighbor_index]
      if neighbor_index not in visited_cities_indexes and distance_to_neighbor < nearest_distance:
        nearest_city_index = neighbor_index
        nearest_distance = distance_to_neighbor

    total_distance += nearest_distance
    current_city = nearest_city_index

  total_distance += distances_matrix[current_city][starting_city]
  solution.append(starting_city)
  return total_distance, solution

def cost_change(distances_matrix, v1, v2, v3, v4):
  return distances_matrix[v1][v3] + distances_matrix[v2][v4] - distances_matrix[v1][v2] - distances_matrix[v3][v4]


def run_2opt_heuristic(distances_matrix, candidate_solution):
  edge_pairs_combinations = list(itertools.combinations(range(0, len(distances_matrix)), 2))
  best = candidate_solution

  has_optimized = True
  while has_optimized:
    has_optimized = False
    for i, j in edge_pairs_combinations:
      v1 = best[i]
      v2 = best[i+1]
      u1 = best[j]
      u2 = best[j+1]

      change = cost_change(distances_matrix, v1, v2, u1, u2)
      if (change < 0):
        novaSolucao = best[:i+1]
        novaSolucao.extend(reversed(best[i+1:j+1]))
        novaSolucao.extend(best[j+1:])
        best = novaSolucao
        has_optimized = True
    
    candidate_solution = best
  return reduce(lambda a, b: a + b, best), best

def run_3opt_heuristic(distances_matrix, candidate_solution):
  edge_pairs_combinations = list(itertools.combinations(range(0, len(distances_matrix)), 3))

  return reduce(lambda a, b: a + b, best), best


def run_vnd_heuristic(distances_matrix):
  solution_cost, solution = run_nn_heuristic(distances_matrix)

  while True:
    # # Call switch most expensive edges
    # solution_cost, solution = run_swith_expensive_edge_heuristic(distances_matrix, solution)
    # Call 2-optonce local optimal is found for most expensive edges
    solution_cost, solution = run_2opt_heuristic(distances_matrix, solution)
    # Call 3-opt once local optimal is found for 2-opt
    solution_cost, solution = run_3opt_heuristic(distances_matrix, solution)
  
  return solution_cost, solution


print('file_name', f'avg_result ({RUNS} runs)', f'avg_time ({RUNS} runs, ms)', sep="\t")
execute_runs("ATT/", True)
# execute_runs("EUC_2D/")
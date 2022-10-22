import math
import random
import time
import os
import itertools
from functools import reduce

RUNS = 1

import os
clear = lambda: os.system('cls')

clear()

def execute_runs(output_file, dir, att = False):
  for filename in os.listdir(dir):
    coords = []
    line_count = 0
    with open(dir+filename) as input_file:
      for line in input_file:
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
    for x in range(RUNS):
      print(f'Executing run {x+1} for {filename}')
      start = time.time()
      result, path = run_vnd_heuristic(distances_matrix)
      # print(path)
      path.sort()
      # print(path)
      elapsed = time.time() - start
      distances_sum += result
      time_sum += elapsed
      # clear()

    output_file.write(f'{filename}\t{round(distances_sum/float(RUNS))}\t{((time_sum/float(RUNS))*float(RUNS) * 1000):.2f}\n') 

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
  visited_cities_indexes = set()
  current_city = starting_city
  solution = []
  total_distance = 0

  while len(visited_cities_indexes) != len(distances_matrix):
    nearest_distance = math.inf
    
    for neighbor_index in range(len(distances_matrix)):
      distance_to_neighbor = distances_matrix[current_city][neighbor_index]
      if neighbor_index != current_city and neighbor_index not in visited_cities_indexes and distance_to_neighbor < nearest_distance:
        nearest_city_index = neighbor_index
        nearest_distance = distance_to_neighbor

    solution.append(current_city)
    visited_cities_indexes.add(current_city)
    current_city = nearest_city_index

  solution.append(starting_city)
  return sum_route_cost(solution, distances_matrix), solution

def cost_change(distances_matrix, v1, v2, v3, v4):
  return distances_matrix[v1][v3] + distances_matrix[v2][v4] - distances_matrix[v1][v2] - distances_matrix[v3][v4]


def run_2opt_heuristic(distances_matrix, candidate_solution):
  edge_pairs_combinations = list(itertools.combinations(range(0, len(distances_matrix)), 2))
  best = candidate_solution

  has_optimized = True
  has_optimized_at_least_once = False
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
        has_optimized_at_least_once = True
    
    candidate_solution = best
  return sum_route_cost(best, distances_matrix), best, has_optimized_at_least_once


def reverse_segment_if_better(distances_matrix, tour, i, j, k):
  def distance(u, v):
    return distances_matrix[u][v]

  A, B, C, D, E, F = tour[i-1], tour[i], tour[j-1], tour[j], tour[k-1], tour[k % len(tour)]

  d0 = distance(A, B) + distance(C, D) + distance(E, F)
  d1 = distance(A, D) + distance(B, F) + distance(E, C)
  d2 = distance(C, F) + distance(B, D) + distance(E, A)
  d3 = distance(B, E) + distance(D, F) + distance(C, A)
  d4 = distance(A, D) + distance(C, F) + distance(B, E)

  if (i - 1) < (k % len(tour)):
      first_segment = tour[k % len(tour):] + tour[:i]
  else:
      first_segment = tour[k % len(tour):i]
  second_segment = tour[i:j]
  third_segment = tour[j:k]

  resulting_delta = 0
  resulting_tour = tour.copy()

  # retornar diferença entre solucao atual e nova solucao
  # > 0 => MELHOROU
  if d0 > d1:
    resulting_delta, resulting_tour = (d0 - d1, (list(reversed(first_segment)) + second_segment + list(reversed(third_segment))))
  elif d0 > d2:
    resulting_delta, resulting_tour = (d0 - d2, (list(reversed(first_segment)) + list(reversed(second_segment)) + third_segment))
  elif d0 > d4:
    resulting_delta, resulting_tour = (d0 - d4, (list(reversed(first_segment)) + list(reversed(second_segment)) + list(reversed(third_segment))))
  elif d0 > d3:
    resulting_delta, resulting_tour = (d0 - d3, (first_segment + list(reversed(second_segment)) + list(reversed(third_segment))))

  return resulting_delta, resulting_tour

  

def run_3opt_heuristic(distances_matrix, candidate_solution):
  edge_triples_combinations = list(itertools.combinations(range(0, len(distances_matrix)), 3))
  best = candidate_solution
  has_optimized = True
  has_optimized_at_least_once = False
  while has_optimized:
    has_optimized = False
    for u, v, w in possible_segments(len(distances_matrix)):
      # source: https://en.wikipedia.org/wiki/3-opt
      # source: http://matejgazda.com/tsp-algorithms-2-opt-3-opt-in-python/

      delta, candidate_solution = reverse_segment_if_better(distances_matrix, candidate_solution, u, v, w)

      if (delta > 0):
        has_optimized = True
        has_optimized_at_least_once = True
        best = candidate_solution
        
  # best.append(best[0])
  return sum_route_cost(best, distances_matrix), best, has_optimized_at_least_once

def possible_segments(N):
    segments = ((i, j, k) for i in range(N) for j in range(i + 2, N-1) for k in range(j + 2, N - 1 + (i > 0)))
    return segments

def sum_route_cost(solucao, matrizDistancias):
    custo = 0
    for i in range(len(solucao)-1):
        custo += matrizDistancias[solucao[i]][solucao[i+1]]
    return custo

def run_vnd_heuristic(distances_matrix):
  solution_cost, solution = run_nn_heuristic(distances_matrix)
  
  while True:
    solution_cost, solution, changed = run_2opt_heuristic(distances_matrix, solution)
    solution_cost, solution, changed = run_3opt_heuristic(distances_matrix, solution)
    if not changed:
      break
  return solution_cost, solution

with open('output.txt', 'w') as output_file:
  output_file.write(f'file_name\tavg_result ({RUNS} runs)\tavg_time ({RUNS} runs, ms)\n')
  execute_runs(output_file, "ATT/", True,)
  # execute_runs(output_file, "EUC_2D/")
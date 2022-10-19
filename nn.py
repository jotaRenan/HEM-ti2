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
  return sum_route_cost(best, distances_matrix), best


def reverse_segment_if_better(distances_matrix, tour, i, j, k):
  def distance(u, v):
    return distances_matrix[u][v]

  A, B, C, D, E, F = tour[i-1], tour[i], tour[j-1], tour[j], tour[k-1], tour[k % len(tour)]
  d0 = distance(A, B) + distance(C, D) + distance(E, F)
  d1 = distance(A, C) + distance(B, D) + distance(E, F)
  d2 = distance(A, B) + distance(C, E) + distance(D, F)
  d3 = distance(A, D) + distance(E, B) + distance(C, F)
  d4 = distance(F, B) + distance(C, D) + distance(E, A)

  if d0 > d1:
    tour[i:j] = reversed(tour[i:j])
    return -d0 + d1
  elif d0 > d2:
    tour[j:k] = reversed(tour[j:k])
    return -d0 + d2
  elif d0 > d4:
    tour[i:k] = reversed(tour[i:k])
    return -d0 + d4
  elif d0 > d3:
    tmp = tour[j:k] + tour[i:j]
    tour[i:k] = tmp
    return -d0 + d3
  return 0

def run_3opt_heuristic(distances_matrix, candidate_solution):
  edge_triples_combinations = list(itertools.combinations(range(0, len(distances_matrix)), 3))
  print(edge_triples_combinations)

  raise 1
  best = candidate_solution
  has_optimized = True
  while has_optimized:
    has_optimized = False
    for i, j, k in edge_triples_combinations:
      for u, v, w in itertools.permutations([best[i], best[j], best[k]]):
        # TODO: calcular corretamente o 3-opt. tudo aqui está incorreto
        # inspiraçao: https://en.wikipedia.org/wiki/3-opt
        # inspiraçao: http://matejgazda.com/tsp-algorithms-2-opt-3-opt-in-python/
        candidate = candidate_solution.copy()
        print(candidate)
        change = reverse_segment_if_better(distances_matrix, candidate, u, v, w)

        if (change < 0):
          has_optimized = True
          # Primeiro aprimorante
          # print('OTIMIZOU')
          print(candidate)
          return sum_route_cost(candidate, distances_matrix), candidate
  return sum_route_cost(best, distances_matrix), best

def sum_route_cost(solucao, matrizDistancias):
    custo = 0
    for i in range(len(solucao)-1):
        custo += matrizDistancias[solucao[i]][solucao[i+1]]
    return custo

def run_vnd_heuristic(distances_matrix):
  solution_cost, solution = run_nn_heuristic(distances_matrix)
  print('  KNN', solution_cost)
  while True:
    solution_cost, solution = run_2opt_heuristic(distances_matrix, solution)
    print('2-OPT', solution_cost)
    # Call 3-opt once local optimal is found for 2-opt
    solution_cost, solution = run_3opt_heuristic(distances_matrix, solution)
    print('3-OPT', solution_cost)
    raise 
  return solution_cost, solution


print('file_name', f'avg_result ({RUNS} runs)', f'avg_time ({RUNS} runs, ms)', sep="\t")
execute_runs("ATT/", True)
# execute_runs("EUC_2D/")
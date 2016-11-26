from memory_profiler import profile
from memory_profiler import memory_usage
# script for Floyd Warshall Algorithm- All Pair Shortest Path to detect shortest path during congestion

## Global variables declaration
INF = 99999999999
NoPath = 999999999
n = 0
distance = 0
nextVertex = 0
hopCount = 0
path_array = 0
load_matrix = 0
actual_edge_delay_matrix = 0
actual_path_delay_matrix = 0

## Miscellaneous methods - handy to print
# fp=open('memory_profiler.log','w+')
# @profile(stream=fp)
def print_edge_values(edge_matrix, actual_edge_matrix, path):
    edges = zip(path[0::1],path[1::1])
    print(" Edge_path, Predicted_edge_length, Actual_edge_length")
    i = 0
    for street in edges:
        print('{:>8} {:>15} {:>20}'.format(str(street), str(edge_matrix[street[0]-1][street[1]-1]), str("INF" if actual_edge_matrix[street[0]-1][street[1]-1] == NoPath else actual_edge_matrix[street[0]-1][street[1]-1])))
        i=i+1
    print("Hop count is ",i)

## Below methods are for calculating the residual values
def calculate_load_residues(load, capacity, input_matrix):
    for u in range(n):
        for v in range(n):
            if(load[u][v]>capacity[u][v]):
                input_matrix[u][v] = INF
                load[u][v] = load[u][v] - capacity[u][v]
                capacity[u][v] = INF


## Actual congestion calculation for the shortest path based on the actual edge delays
# @profile(stream=fp)
def actual_path_delay(shortest_path, actual_edge_delay, output):
    for u in range(n):
        for v in range(n):
            route = shortest_path[u][v]
            street_map = zip(route[0::1], route[1::1])
            for i in street_map:
                if (actual_edge_delay[i[0] - 1][i[1] - 1] == NoPath):
                    output[u][v] = NoPath
                    break
                else:
                    output[u][v] += actual_edge_delay[i[0]-1][i[1]-1]
                    output[u][v] = round(output[u][v],2)

## Formula for congestion calculation -> G[i,j] = (C[i][j]+1/(C[i][j]+1-L[i][j])*E[i][j])
# @profile(stream=fp)
def congestion_formula(i,j, c):
    if(i==j):
        actual_edge_delay_matrix[i][j]=0
    elif(edge_weights[i][j]==INF):
        actual_edge_delay_matrix[i][j]=INF
    else:
        congestion = ((c+1)/(c+1 - load_matrix[i][j]))*edge_weights[i][j]
        if (congestion>0):
            actual_edge_delay_matrix[i][j] = round(congestion, 2)
        else:
            actual_edge_delay_matrix[i][j] = NoPath



## Actual_edge_delay() based on the congestion using the congestion formula Congestion(G) = (Capacity(C)+1/Capacity(C)+1-Load(L))*EdgeWeight(E)
# @profile(stream=fp)
def actual_edge_delay(capacity):
    for u in range(n):
        for v in range(n):
            congestion_formula(u,v,capacity[u][v])


## Below methods will calculate the load on each edge
# @profile(stream=fp)
def calculate_load(edge_matrix, flow_matrix, output_matrix):
    for u in range(n):
        for v in range(n):
             if(edge_matrix[u][v] == INF):
                 output_matrix[u][v] = INF
             if(1==1):
                route = path_array[u][v]
                directions = zip(route[0::1], route[1::1])
                for i in directions:
                    output_matrix[i[0]-1][i[1]-1] += flow_matrix[u][v]


## Below methods are helpful in calculating and populating the distance matrix, hop count matrix, shortest path matrix
# floydWarshall() -> calculates shortest distances and updates distance matrix
# printShortestDistance() -> prints distances in pretty format
# printShortestPath() -> prints shortest path and updates path_array matrix using get_path() method
# get_path() -> calculates and updates the path_array matrix and hop count matrix
# printHopCount() - prints hop count in pretty format

# @profile(stream=fp)
def printShortestDistance(distGraph):
    print('\n'.join([''.join(['{:13}'.format('{:>13}'.format("na") if item==INF else '{:>13}'.format("INF") if item==NoPath else item) for item in row])
                     for row in distGraph]))
    print("\n")
    print("\n")


# @profile(stream=fp)
def get_path(origin, dest):
  """
  Reconstruct shortest path from each of u to v using the predecessor matrix passed as input
  This method is not recursive to avoid reach  of call stack limit when input is a large matrix
  """
  u = origin - 1
  v = dest - 1

  path_list = [origin]
  while 1:
    predecessor = nextVertex[origin - 1][dest - 1] + 1
    if predecessor == origin:
        path_array[u][v] = path_list
        return map(str, path_list)
    path_list.append(int(predecessor))
    hopCount[u][v] = hopCount[u][v]+1
    origin = predecessor


# @profile(stream=fp)
def printShortestPath():
    print("Floyd Warshall Graph Paths")
    for u, i in enumerate(range(0, n), 1):
        for v, j in enumerate(range(0, n), 1):
            print(('(')+','.join(get_path(u,v))+')', end="      ")
        print("\n")
    print("\n")


# @profile(stream=fp)
def printHopCount():
    print("Hop count:")
    for u in range(n):
        for v in range(n):
            print(hopCount[u][v], end="  ")
        print(" ")
    print("\n")


# @profile(stream=fp)
def floydWarshall(EdgeGraph):
    # Initialize the edge weights to distances.
    for u in range(n):
        for v in range(n):
            distance[u][v] = EdgeGraph[u][v]
            nextVertex[u][v] = v

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if distance[i][k] + distance[k][j] < distance[i][j]:
                    distance[i][j] = round(distance[i][k] + distance[k][j],2)
                    distance[i][j] = round(distance[i][j])
                    nextVertex[i][j] = nextVertex[i][k]

def output_loop(iteration_value):
    while(iteration_value>0):
        iteration_value = iteration_value - 1

        floydWarshall(edge_weights)
        print("\nFloyd Warshall shortest path distances:")
        printShortestDistance(distance)
        printShortestPath()
        printHopCount()
        calculate_load(edge_weights, flow_matrix, load_matrix)
        print("\nLoad_matrix:")
        printShortestDistance(load_matrix)
        actual_edge_delay(capacity_matrix)
        print("\nActual edge delay matrix:")
        printShortestDistance(actual_edge_delay_matrix)
        actual_path_delay(path_array, actual_edge_delay_matrix, actual_path_delay_matrix)
        print("\nActual path delay matrix:")
        printShortestDistance(actual_path_delay_matrix)

        print("The shortest predicted path length between a:{} and b:{} is {}".format(a, b, distance[a - 1][
                        b - 1]))
        print("The actual path length between a:{} and b:{} is {}".format(a, b, "INF" if
                    actual_path_delay_matrix[a - 1][b - 1] == NoPath else actual_path_delay_matrix[a - 1][b - 1]))
        print("The actual path length and the predicted path length are within",
                          round(((actual_path_delay_matrix[a - 1][
                                      b - 1] - distance[a - 1][
                                      b - 1]) /
                                 actual_path_delay_matrix[a - 1][
                                     b - 1] * 100), 2),
                          "of each other")

        print("\nThe shortest path between a:{} and b:{} is {}".format(a, b, path_array[a - 1][b - 1]))
        print("The predicted edge lengths from first and last edge on the path are:")
        print_edge_values(edge_weights, actual_edge_delay_matrix, path_array[a - 1][b - 1])


# Main method - Execution begins from here
if __name__ == '__main__':
    file = open('input.txt')
    iteration = 2
    raw_lines = file.read().splitlines()
    lines = list(line for line in raw_lines if line)
    first_line = lines[0].strip().split(',')
    n = int(first_line[0])
    edge_weights = [[0 if row==col else INF for row in range(n)] for col in range(n)]
    flow_matrix = [[0 if row==col else INF for row in range(n)] for col in range(n)]
    capacity_matrix = [[0 if row==col else INF for row in range(n)] for col in range(n)]
    # a = int(first_line[1])
    # b = int(first_line[2])
    a=2
    b=1

# Initial empty matrices declaration based on the range value
    distance = [[0 for row in range(n)] for col in range(n)]
    nextVertex = [[0 for row in range(n)] for col in range(n)]
    hopCount = [[0 for row in range(n)] for col in range(n)]
    path_array = [[0 for row in range(n)] for col in range(n)]
    load_matrix = [[0 for row in range(n)] for col in range(n)]
    actual_edge_delay_matrix = [[0 for row in range(n)] for col in range(n)]
    actual_path_delay_matrix = [[0 for row in range(n)] for col in range(n)]

# Read EdgeWeights, Flow, Capacity values from the input file
    for raw_line in lines:
        line = raw_line.strip()
        if(line.startswith('E')):
            items = line.strip('E').split(',')
            u = int(items[1].strip())
            v = int(items[2].strip())
            value = int(items[3].strip())
            edge_weights[u-1][v-1] = value
        elif(line.startswith('F')):
            items = line.strip('F').split(',')
            u = int(items[1].strip())
            v = int(items[2].strip())
            value = int(items[3].strip())
            flow_matrix[u-1][v-1] = value
        elif (line.startswith('C')):
            items = line.strip('C').split(',')
            u = int(items[1].strip())
            v = int(items[2].strip())
            value = int(items[3].strip())
            capacity_matrix[u-1][v-1] = value

    #Main Loop
    floydWarshall(edge_weights)
    print("\nFloyd Warshall shortest path distances:")
    printShortestDistance(distance)
    printShortestPath()
    printHopCount()
    calculate_load(edge_weights, flow_matrix, load_matrix)
    print("\nLoad_matrix:")
    printShortestDistance(load_matrix)
    actual_edge_delay(capacity_matrix)
    print("\nActual edge delay matrix:")
    printShortestDistance(actual_edge_delay_matrix)
    actual_path_delay(path_array, actual_edge_delay_matrix, actual_path_delay_matrix)
    print("\nActual path delay matrix:")
    printShortestDistance(actual_path_delay_matrix)

    print("The shortest predicted path length between a:{} and b:{} is {}".format(a, b, distance[a - 1][
        b - 1]))
    print("The actual path length between a:{} and b:{} is {}".format(a, b, "INF" if
    actual_path_delay_matrix[a - 1][b - 1] == NoPath else actual_path_delay_matrix[a - 1][b - 1]))
    print("The actual path length and the predicted path length are within",
          round(((actual_path_delay_matrix[a - 1][
                      b - 1] - distance[a - 1][
                      b - 1]) /
                 actual_path_delay_matrix[a - 1][
                     b - 1] * 100), 2),
          "of each other")

    print("\nThe shortest path between a:{} and b:{} is {}".format(a, b, path_array[a - 1][b - 1]))
    print("The predicted edge lengths from first and last edge on the path are:")
    print_edge_values(edge_weights, actual_edge_delay_matrix, path_array[a - 1][b - 1])


    #Iteration:1
    actual_edge_delay_matrix = [[INF if item > 99999999 else item for item in row] for row in actual_edge_delay_matrix]
    edge_weights = actual_edge_delay_matrix
    distance = [[0 for row in range(n)] for col in range(n)]
    nextVertex = [[0 for row in range(n)] for col in range(n)]
    hopCount = [[0 for row in range(n)] for col in range(n)]
    path_array = [[0 for row in range(n)] for col in range(n)]
    load_matrix = [[0 for row in range(n)] for col in range(n)]
    actual_edge_delay_matrix = [[0 for row in range(n)] for col in range(n)]
    actual_path_delay_matrix = [[0 for row in range(n)] for col in range(n)]

    print("Edge_Matrix:", edge_weights)
    print("distance_Matrix: ", distance)
    print("next_Vertex_Matrix:", nextVertex)
    print("hopCount_Matrix:", hopCount)
    print("path_array:", path_array)
    print("actual_edge_delay_matrix: ", actual_edge_delay_matrix)
    print("actual_edge_path_matrix: ", actual_path_delay_matrix)

    floydWarshall(edge_weights)
    print("\nFloyd Warshall shortest path distances:")
    printShortestDistance(distance)
    printShortestPath()
    printHopCount()
    calculate_load(edge_weights, flow_matrix, load_matrix)
    print("\nLoad_matrix:")
    printShortestDistance(load_matrix)
    actual_edge_delay(capacity_matrix)
    print("\nActual edge delay matrix:")
    printShortestDistance(actual_edge_delay_matrix)
    actual_path_delay(path_array, actual_edge_delay_matrix, actual_path_delay_matrix)
    print("\nActual path delay matrix:")
    printShortestDistance(actual_path_delay_matrix)

    print("The shortest predicted path length between a:{} and b:{} is {}".format(a, b, distance[a - 1][
        b - 1]))
    print("The actual path length between a:{} and b:{} is {}".format(a, b, "INF" if
    actual_path_delay_matrix[a - 1][b - 1] == NoPath else actual_path_delay_matrix[a - 1][b - 1]))
    print("The actual path length and the predicted path length are within",
          round(((actual_path_delay_matrix[a - 1][
                      b - 1] - distance[a - 1][
                      b - 1]) /
                 actual_path_delay_matrix[a - 1][
                     b - 1] * 100), 2),
          "of each other")

    print("\nThe shortest path between a:{} and b:{} is {}".format(a, b, path_array[a - 1][b - 1]))
    print("The predicted edge lengths from first and last edge on the path are:")
    print_edge_values(edge_weights, actual_edge_delay_matrix, path_array[a - 1][b - 1])

    # Iteration:2
    actual_edge_delay_matrix = [[INF if item > 99999999 else item for item in row] for row in actual_edge_delay_matrix]
    edge_weights = actual_edge_delay_matrix
    distance = [[0 for row in range(n)] for col in range(n)]
    nextVertex = [[0 for row in range(n)] for col in range(n)]
    hopCount = [[0 for row in range(n)] for col in range(n)]
    path_array = [[0 for row in range(n)] for col in range(n)]
    load_matrix = [[0 for row in range(n)] for col in range(n)]
    actual_edge_delay_matrix = [[0 for row in range(n)] for col in range(n)]
    actual_path_delay_matrix = [[0 for row in range(n)] for col in range(n)]

    print("Edge_Matrix:", edge_weights)
    print("distance_Matrix: ", distance)
    print("next_Vertex_Matrix:", nextVertex)
    print("hopCount_Matrix:", hopCount)
    print("path_array:", path_array)
    print("actual_edge_delay_matrix: ", actual_edge_delay_matrix)
    print("actual_edge_path_matrix: ", actual_path_delay_matrix)

    floydWarshall(edge_weights)
    print("\nFloyd Warshall shortest path distances:")
    printShortestDistance(distance)
    printShortestPath()
    printHopCount()
    calculate_load(edge_weights, flow_matrix, load_matrix)
    print("\nLoad_matrix:")
    printShortestDistance(load_matrix)
    actual_edge_delay(capacity_matrix)
    print("\nActual edge delay matrix:")
    printShortestDistance(actual_edge_delay_matrix)
    actual_path_delay(path_array, actual_edge_delay_matrix, actual_path_delay_matrix)
    print("\nActual path delay matrix:")
    printShortestDistance(actual_path_delay_matrix)

    print("The shortest predicted path length between a:{} and b:{} is {}".format(a, b, distance[a - 1][
        b - 1]))
    print("The actual path length between a:{} and b:{} is {}".format(a, b, "INF" if
    actual_path_delay_matrix[a - 1][b - 1] == NoPath else actual_path_delay_matrix[a - 1][b - 1]))
    print("The actual path length and the predicted path length are within",
          round(((actual_path_delay_matrix[a - 1][
                      b - 1] - distance[a - 1][
                      b - 1]) /
                 actual_path_delay_matrix[a - 1][
                     b - 1] * 100), 2),
          "of each other")

    print("\nThe shortest path between a:{} and b:{} is {}".format(a, b, path_array[a - 1][b - 1]))
    print("The predicted edge lengths from first and last edge on the path are:")
    print_edge_values(edge_weights, actual_edge_delay_matrix, path_array[a - 1][b - 1])

    # Iteration:3
    actual_edge_delay_matrix = [[INF if item > 99999999 else item for item in row] for row in actual_edge_delay_matrix]
    edge_weights = actual_edge_delay_matrix
    distance = [[0 for row in range(n)] for col in range(n)]
    nextVertex = [[0 for row in range(n)] for col in range(n)]
    hopCount = [[0 for row in range(n)] for col in range(n)]
    path_array = [[0 for row in range(n)] for col in range(n)]
    load_matrix = [[0 for row in range(n)] for col in range(n)]
    actual_edge_delay_matrix = [[0 for row in range(n)] for col in range(n)]
    actual_path_delay_matrix = [[0 for row in range(n)] for col in range(n)]

    print("Edge_Matrix:", edge_weights)
    print("distance_Matrix: ", distance)
    print("next_Vertex_Matrix:", nextVertex)
    print("hopCount_Matrix:", hopCount)
    print("path_array:", path_array)
    print("actual_edge_delay_matrix: ", actual_edge_delay_matrix)
    print("actual_edge_path_matrix: ", actual_path_delay_matrix)

    floydWarshall(edge_weights)
    print("\nFloyd Warshall shortest path distances:")
    printShortestDistance(distance)
    printShortestPath()
    printHopCount()
    calculate_load(edge_weights, flow_matrix, load_matrix)
    print("\nLoad_matrix:")
    printShortestDistance(load_matrix)
    actual_edge_delay(capacity_matrix)
    print("\nActual edge delay matrix:")
    printShortestDistance(actual_edge_delay_matrix)
    actual_path_delay(path_array, actual_edge_delay_matrix, actual_path_delay_matrix)
    print("\nActual path delay matrix:")
    printShortestDistance(actual_path_delay_matrix)

    print("The shortest predicted path length between a:{} and b:{} is {}".format(a, b, distance[a - 1][
        b - 1]))
    print("The actual path length between a:{} and b:{} is {}".format(a, b, "INF" if
    actual_path_delay_matrix[a - 1][b - 1] == NoPath else actual_path_delay_matrix[a - 1][b - 1]))
    print("The actual path length and the predicted path length are within",
          round(((actual_path_delay_matrix[a - 1][
                      b - 1] - distance[a - 1][
                      b - 1]) /
                 actual_path_delay_matrix[a - 1][
                     b - 1] * 100), 2),
          "of each other")

    print("\nThe shortest path between a:{} and b:{} is {}".format(a, b, path_array[a - 1][b - 1]))
    print("The predicted edge lengths from first and last edge on the path are:")
    print_edge_values(edge_weights, actual_edge_delay_matrix, path_array[a - 1][b - 1])

    # Recalculating the Shortest paths using G[i,j] as the weights
    #calculate_load_residues(load_matrix, capacity_matrix, edge_weights)
import numpy

# script for Floyd Warshall Algorithm- All Pair Shortest Path
INF = 9999999
n = 0
distance = 0
nextVertex = 0
hopCount = 0
path_array = 0
load_matrix = 0
actual_edge_delay_matrix = 0
actual_path_delay_matrix = 0


## actual congestion calculation for the shortest path based on the actual edge delays
def actual_path_delay(shortest_path, actual_edge_delay, output):
    for u in range(n):
        for v in range(n):
            route = shortest_path[u][v]
            street_map = zip(route[0::1], route[1::1])
            for i in street_map:
                output[u][v] += actual_edge_delay[i[0]-1][i[1]-1]


## formula for congestion calculation
def congestion_formula(i,j, c):
    if(edge_weights[i][j]==INF):
        actual_edge_delay_matrix[i][j]=INF
    else:
        actual_edge_delay_matrix[i][j] = round(((c+1)/(c+1 - load_matrix[i][j]))*edge_weights[i][j], 2)


## actual_edge_delay()
def actual_edge_delay(capacity):
    for u in range(n):
        for v in range(n):
            congestion_formula(u,v,capacity[u][v])


## Below methods will calculate the load on each edge
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


## actual path delay/cost calculation based on the shortest path with actual delay matrix
# def actual_path_delay():
#     calculate_load(actual_edge_delay_matrix, flow_matrix)
# print(actual_path_delay_matrix)

## Below methods are helpful in calculating and populating the distance matrix, hop count matrix, shortest path matrix
# floydWarshall() -> calculates shortest distances and updates distance matrix
# printShortestDistance() -> prints distances in pretty format
# printShortestPath() -> prints shortest path and updates path_array matrix using get_path() method
# get_path() -> calculates and updates the path_array matrix and hop count matrix
# printHopCount() - prints hop count in pretty format

def printShortestDistance(distGraph):
    print('\n'.join([''.join(['{:8}'.format('{:>8}'.format("na") if item==INF else item) for item in row])
                     for row in distGraph]))
    print("\n")
    print("\n")


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


def printShortestPath():
    print("Floyd Warshall Graph Paths")
    for u, i in enumerate(range(0, n), 1):
        for v, j in enumerate(range(0, n), 1):
            print(('(')+','.join(get_path(u,v))+')', end="      ")
        print("\n")
    print("\n")


def printHopCount():
    print("Hop count:")
    for u in range(n):
        for v in range(n):
            print(hopCount[u][v], end="  ")
        print(" ")
    print("\n")


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
                    distance[i][j] = distance[i][k] + distance[k][j]
                    nextVertex[i][j] = nextVertex[i][k]


if __name__ == '__main__':

    n = 6
    # file = open('input.txt')
    # raw_lines = file.read().splitlines()
    # lines = list(line for line in raw_lines if line)
    # first_line = lines[0].strip().split(',')
    # n = int(first_line[0])
    edge_weights = [[0 if row==col else INF for row in range(n)] for col in range(n)]
    flow_matrix = [[0 if row==col else INF for row in range(n)] for col in range(n)]
    capacity_matrix = [[0 if row==col else INF for row in range(n)] for col in range(n)]
    # a = int(first_line[1])
    # b = int(first_line[2])

# Initial empty matrices declaration
    distance = [[0 for row in range(n)] for col in range(n)]
    nextVertex = [[0 for row in range(n)] for col in range(n)]
    hopCount = [[0 for row in range(n)] for col in range(n)]
    path_array = [[0 for row in range(n)] for col in range(n)]
    load_matrix = [[0 for row in range(n)] for col in range(n)]
    actual_edge_delay_matrix = [[0 for row in range(n)] for col in range(n)]
    actual_path_delay_matrix = [[0 for row in range(n)] for col in range(n)]

    # for raw_line in lines:
    #     line = raw_line.strip()
    #     if(line.startswith('E')):
    #         items = line.strip('E').split(',')
    #         u = int(items[1].strip())
    #         v = int(items[2].strip())
    #         value = int(items[3].strip())
    #         edge_weights[u-1][v-1] = value
    #     elif(line.startswith('F')):
    #         items = line.strip('F').split(',')
    #         u = int(items[1].strip())
    #         v = int(items[2].strip())
    #         value = int(items[3].strip())
    #         flow_matrix[u-1][v-1] = value
    #     elif (line.startswith('C')):
    #         items = line.strip('C').split(',')
    #         u = int(items[1].strip())
    #         v = int(items[2].strip())
    #         value = int(items[3].strip())
    #         capacity_matrix[u-1][v-1] = value

    # Edge weight input filling Column -> Rows values
    edge_weights = [[0, 7, INF, 7, INF, 9],
                    [INF, 0, 5, INF, 10, 3],
                    [9, 10, 0, 8, 4, 6],
                    [9, 4, 2, 0, INF, INF],
                    [3, 5, 10, 10, 0, INF],
                    [INF, 5, 8, 10, INF, 0]]

    flow_matrix =  [[0, 9, 11, 12, 8, 12],
                    [18, 0, 15, 10, 17, 18],
                    [17, 18, 0, 14, 10, 10],
                    [17, 8, 10, 0, 17, 18],
                    [15, 9, 12, 14, 0, 16],
                    [18, 16, 15, 8, 9, 0]]

    capacity_matrix =  [[0, 13, INF, 33, INF, 20],
                        [INF, 0, 67, INF, 5, 55],
                        [5, 5, 0, 32, 134, 17],
                        [23, 34, 55, 0, INF, INF],
                        [68, 47, 20, 14, 0, INF],
                        [INF, 16, 44, 16, INF, 0]]

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




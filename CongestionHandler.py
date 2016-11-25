# script for Floyd Warshall Algorithm- All Pair Shortest Path
INF = 9999999

distance = [[0 for row in range(6)] for col in range(6)]
nextVertex = [[0 for row in range(6)] for col in range(6)]
hopCount = [[0 for row in range(6)] for col in range(6)]
path_array = [[0 for row in range(6)] for col in range(6)]

load_matrix = [[0 for row in range(6)] for col in range(6)]

#def calculate_load(flow_matrix):




## Below methods are helpful in calculating and populating the distance matrix, hop count matrix, shortest path matrix
# floydWarshall() -> calculates shortest distances and updates distance matrix
# printShortestDistance() -> prints distances in pretty format
# printShortestPath() -> prints shortest path and updates path_array matrix using get_path() method
# get_path() -> calculates and updates the path_array matrix and hop count matrix
# printHopCount() - prints hop count in pretty format

def printShortestDistance(distGraph):
    print("Floyd Warshall Graph Distances:")
    print('\n'.join([''.join(['{:5}'.format(item) for item in row])
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
    for u, i in enumerate(range(0, 6), 1):
        for v, j in enumerate(range(0, 6), 1):
            print(('(')+','.join(get_path(u,v))+')', end="      ")
        print("\n")
    print("\n")


def printHopCount():
    print("Hop count:")
    for u in range(6):
        for v in range(6):
            print(hopCount[u][v], end="  ")
        print(" ")


def floydWarshall(EdgeGraph):
    # Initialize the edge weights to distances.
    for u in range(6):
        for v in range(6):
            distance[u][v] = EdgeGraph[u][v]
            nextVertex[u][v] = v

    for k in range(6):
        for i in range(6):
            for j in range(6):
                if distance[i][k] + distance[k][j] < distance[i][j]:
                    distance[i][j] = distance[i][k] + distance[k][j]
                    nextVertex[i][j] = nextVertex[i][k]
    printShortestDistance(distance)
    printShortestPath()
    printHopCount()

if __name__ == '__main__':
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

    floydWarshall(edge_weights)
    calculate_load(flow_matrix)




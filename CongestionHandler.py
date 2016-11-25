# script for Floyd Warshall Algorithm- All Pair Shortest Path
INF = 9999999

distance = [[0 for row in range(6)] for col in range(6)]
nextVertex = [[0 for row in range(6)] for col in range(6)]

def printShortestDistance(distGraph):
    print("Floyd Warshall Graph Distances")
    print('\n'.join([''.join(['{:5}'.format(item) for item in row])
                     for row in distGraph]))


def get_path(origin, dest):
  """
  Reconstruct shortest path from origin to destination using the predecessor matrix passed as input
  This method is not recursive to avoid reach  of call stack limit when input is a large matrix
  """
  path_list = [origin]
  while 1:
    predecessor = nextVertex[origin - 1][dest - 1] + 1
    if predecessor == origin:
      return map(str, path_list)
    path_list.append(int(predecessor))
    origin = predecessor


def printShortestPath():
    shortestPath = [[0 for i in range(6)] for j in range(6)]
    print(" ")
    print("Floyd Warshall Graph Paths")
    for u, i in enumerate(range(0, 6), 1):
        for v, j in enumerate(range(0, 6), 1):
            print(','.join(get_path(u, v)), end="      ")
        print("\n")


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

if __name__ == '__main__':
    # Edge weight input filling Column -> Rows values
    edge_weights = [[0, 7, INF, 7, INF, 9],
                    [INF, 0, 5, INF, 10, 3],
                    [9, 10, 0, 8, 4, 6],
                    [9, 4, 2, 0, INF, INF],
                    [3, 5, 10, 10, 0, INF],
                    [INF, 5, 8, 10, INF, 0]]

    floydWarshall(edge_weights)


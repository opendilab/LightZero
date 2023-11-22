"""
A搜索算法是一种在图形中找到最短路径的算法，常用在路线规划和游戏等领域。A算法的主要特点是：它使用了启发式函数来估计从起点到终点的最佳路径，从而提高搜索效率。

A*算法的步骤如下：

在开始时，只有起点在开放列表（open list）中。这个列表包含了待评估的节点。
在每一步，算法选择开放列表中具有最低评估函数值的节点，并检查这个节点是不是目标节点。如果是，那么算法就完成了。否则，这个节点就被移到关闭列表（closed list）中，表示已经被评估过了。
然后，对选中节点的所有邻居进行评估。如果邻居节点不在开放列表中，就把它添加进去，并且把当前节点设置为这个邻居节点的父节点。如果邻居节点已经在开放列表中，就检查经过当前节点是否能得到一条路径更短。如果可以，就更新这个邻居节点的父节点为当前节点。
重复第2和第3步，直到找到目标节点，或者开放列表为空，表示图形中不存在一条从起点到终点的路径。
下面是一个使用A*算法找到迷宫路径的Python代码示例：

注意：graph 是一个实现了方法 neighbors(node) 和 cost(a, b) 的对象，
其中 neighbors(node) 返回节点的邻居列表，cost(a, b) 返回从 a 到 b 的代价。
start 和 goal 是图中的节点，表示起点和终点。

在实际使用时，你需要根据你的具体问题来定义 graph，start 和 goal，并实现 neighbors(node) 和 cost(a, b) 方法。
"""
import heapq


class Graph:
    def __init__(self):
        self.edges = {}

    def neighbors(self, id):
        return self.edges[id]

    def cost(self, a, b):
        return 1


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return not self.elements

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


def heuristic(a, b):
    return abs(b[0] - a[0]) + abs(b[1] - a[1])


def a_star(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            break

        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(goal, next)
                frontier.put(next, priority)
                came_from[next] = current

    return came_from, cost_so_far


# Define the graph
graph = Graph()
graph.edges = {
    (0, 0): [(0, 1), (1, 0)],
    (0, 1): [(0, 0), (0, 2)],
    (0, 2): [(0, 1), (1, 2)],
    (1, 0): [(0, 0), (1, 1)],
    (1, 1): [(1, 0), (1, 2)],
    (1, 2): [(0, 2), (1, 1)],
}

start, goal = (0, 0), (1, 2)
came_from, cost_so_far = a_star(graph, start, goal)

# Print the path
current = goal
path = []
while current != start:
    path.append(current)
    current = came_from[current]
path.append(start)  # optional
path.reverse()  # optional

print("Path found: ", path)
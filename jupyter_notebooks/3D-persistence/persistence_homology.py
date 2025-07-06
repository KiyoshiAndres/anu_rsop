def compute_components(vertices: list, filtration: list):
    # Union-Find (Disjoint Set) Implementation with Detailed Debugging
    uf = UnionFind(len(vertices))
    connected_components = []
    merger = []
    
    for height, section in filtration.items():
        section_vertices = section['points']
        horizontal_edges = section['horizontal_edges']
        horizontal_components = compute_horizontal_step(section_vertices, horizontal_edges)
        
    
def compute_horizontal_step(vertices: list, horizontal_edges: list) -> list:
    '''Input: List of vertices of height n, and list of horizontal edges of height n.
    Output: TODO
    '''
    components = []
    return components   
    

def compute_vertical_step(previous_components: list, current_components: list, angled_edges: list) -> list:
    '''Input: List of vertices of height n, and list of horizontal edges of height n.
    Output: TODO
    Prints: 
        New Connected Components (Subset of current__components):
        Merged Components (Subset of previous_components):
    '''
    for edge in angled_edges:
        # UnionFind merge current connected component with previous_connected_component
        # If current connected component was already merged with that edge do nothing
        # If current connected component wasn't already merged with that edge, add root to a set
        edge
        
    
    new_connected_components = []
    merged_components = []
    components = []
    return components    
    
    
class UnionFind:
    def __init__(self, vertices: list[dict]):
        """
        Initialize the union-find structure with 'n' elements.
        Each element starts as its own parent, meaning each is a separate set.
        'rank' is the vertex index for union by rank.
        """
        self.parent = list(range(len(vertices)))
        self.rank = list(range(len(vertices)))

    def find(self, x):
        """
        The 'find' function locates the root (representative) of the set
        containing 'x'. It also applies path compression, so that each visited
        node directly points to the root, which speeds up future calls.
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """
        The 'union' function merges the sets containing 'x' and 'y'.
        It first finds the roots of both nodes, and if they are different,
        it attaches the tree with lower rank to the tree with higher rank.
        If the ranks are equal, it arbitrarily chooses one as the new root
        and increments its rank.
        """
        rootX = self.find(x)
        rootY = self.find(y)
        rankX = self.rank[rootX]
        rankY = self.rank[rootY]

        if rootX == rootY:
            return {x: {'root': rootX, 'rank': rankX}, y: {'root': rootY, 'rank': rankY}}

        # Union by rank: attach the smaller tree under the larger tree
        if rankX > rankY:
            self.parent[rootX] = rootY
        elif rankX < rankY:
            self.parent[rootY] = rootX
        else:
            # If ranks are equal, default to the first.
            self.parent[rootY] = rootX

        return {x: {'root': rootX, 'rank': rankX}, y: {'root': rootY, 'rank': rankY}}


# 




# Preprocessing

def group_events_by_height(points, edges):
    """
    Input is a set of points, and a set of edges.
    Returns a dict of the form:
    
    """
    events_by_height = {}
    for point in points:
        h = point['height']
        events_by_height.setdefault(h, {'points': [], 'horizontal_edges': [], 'vertical_edges': []})
        events_by_height[h]['points'].append(point)

    for edge in edges:
        h = max(edge['height'])
        events_by_height.setdefault(h, {'points': [], 'horizontal_edges': [], 'vertical_edges': []})
        if min(edge['height'])==max(edge['height']):
            events_by_height[h]['horizontal_edges'].append(edge)
        else:
            events_by_height[h]['vertical_edges'].append(edge)
    return events_by_height



def process_graph(vertices, edges, direction):
    """
        The input are vertices and edges and a direction.
        
        The output is a graph ordered by height, and by x,y,z. The normal vectors are replaced with the sign.
    """
    processed_graph = order_graph(vertices, edges)
    graph = [processed_graph['vertices'], processed_graph['edges']]
    signed_graph = obtain_sign(graph, direction)
    return {'signed_graph': signed_graph, 'index_translation': processed_graph['index_translation']}

def subdivide_edges(edges: list) -> list:
    '''Input: List of edges formated as 
    edge = ['vertices': [index_i, index_j], 'height': [height_i, height_j], 'n': n]
    Output: Partitions the edges for processing in two steps.
    A list containing two lists of edges, the first entry is horizontal edges.
    '''
    horizontal_edges = []
    angled_edges = []
    for edge in edges:
        if min(edge['height'])==max(edge['height']):
            horizontal_edges.append(edge)
        else:
            angled_edges.append(edge)
    return [horizontal_edges, angled_edges]

# obtain_sign, sign, and order_graph are helper functions for process_graph. 

def obtain_sign(graph, direction: list) -> list:
    points, edges = graph
    signed_points = []
    signed_edges = []
    for point in points:
        point['sign'] = sign(point['normal'], direction)
        del point['normal']
        signed_points.append(point)
    for e in edges:
        signed_edges.append({'vertices': e['vertices'], 'height': e['height'], 'sign': sign(e['n'], direction)})
    return [signed_points, signed_edges]

def sign(v_1,v_2):
    product = v_1[0] * v_2[0] +  v_1[1] * v_2[1] + v_1[2] * v_2[2]
    sign = 0
    if product > 0:
        sign = 1
    elif product < 0:
        sign = -1
    return sign

def order_graph(vertices, edges):
    """
        The input are vertices and edges.
        {'coordinates': [i, j, k], 'original_index': idx, 'new_index': idx, 'height': h, 'normal': n}
        {'vertices': [e, l], 'height': [h_0,h_1], 'n': n}
        
        The output is a graph ordered by height, and by x,y,z.
    """

    # Step 1: Sort the vertices
    sorted_vertices = sorted(
        vertices,
        key=lambda v: (v['height'], v['coordinates'][0], v['coordinates'][1], v['coordinates'][2])
    )

    # Step 2: Relabel the vertices
    original_to_new_index = {}
    for new_index, vertex in enumerate(sorted_vertices):
        original_index = vertex['original_index']
        original_to_new_index[original_index] = new_index
        vertex['new_index'] = new_index


    
    # Step 3: Update the edges
    for edge in edges:
        # Map old indices to new indices and sort them within the edge
        new_indices = [original_to_new_index[vi] for vi in edge['vertices']]
        edge['vertices'] = new_indices
    # Step 4: Sort the edges
    sorted_edges = sorted(
        edges,
        key=lambda e: (max(e['height']), min(e['vertices']))
    )
    
    output_vertices = [ v for v in sorted_vertices ]
    output_edges = [ {'vertices': e['vertices'], 'height': e['height'], 'n': e['n'] } for e in sorted_edges ]
    return {'vertices': output_vertices, 'edges': output_edges, 'index_translation': original_to_new_index}

def height_of_vertex(direction: list, point: list):
    height = 0
    for n in list(range(3)):
        height_squared = direction[n] * point[n]
        height += height_squared
    return height


# Formatting Edges and Vertices

def append_height_vertices(direction: list[int, int, int], vertices: list):
    '''Input:
        List of vertices [
    
    '''
    new_vertices = []
    for vertex in vertices:
        height = height_of_vertex(direction, vertex[0])
        new_vertices.append([vertex[0],height, vertex[1]])
    return new_vertices

def format_vertices(vertices: list) -> list:
    # Input: [coord, height, vector n]
    new_vertices = []
    n = 0
    for vertex in vertices:
        new_vertices.append({'coordinates': vertex[0], 
                             'original_index': n, 
                             'new_index': None,
                             'height': vertex[1],
                             'normal': vertex[2]
                            })
        n += 1
    return new_vertices


def format_edges(points: list, edges: list) -> list:
    # Input: []
    formatted_edges = []
    for edge in edges:
        l_vertex_index = edge[0][0]
        r_vertex_index = edge[0][1]
        l_height = points[l_vertex_index]['height']
        r_height = points[r_vertex_index]['height']
        formatted_edges.append({'vertices': [l_vertex_index, r_vertex_index], 'height': [l_height, r_height], 'n': edge[1]})
    return formatted_edges


# Helper Functions

def reindex_edges(edges: list[int, int]) -> list[int, int]:
    '''
        Input: list of edges [m,n] where m,n >= 1 (indexing starts at 1)
        Output: same list of edges [m-1,n-1]
    '''
    reindexed_edges = []
    for edge in edges:
        reindexed_edges.append([edge[0] - 1, edge[1] - 1])
    return reindexed_edges





#deprecated

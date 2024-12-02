def group_events_by_height(points, edges):
    """
    Input is a set of points, and a set of edges.
    Returns a dict of the form:
    
    """
    events_by_height = {}
    for i, h, n in points:
        events_by_height.setdefault(h, {'points': [], 'edges': []})
        events_by_height[h]['points'].append((i, h, n))

    for (i_j, h, n) in edges:
        events_by_height.setdefault(h, {'points': [], 'edges': []})
        events_by_height[h]['edges'].append((i_j, h, n))
        
    return events_by_height


def compute_critical_points(graph):
    """
    This accepts a graph of the shape [points, edges]
    Where points and edges are lists of the shape:
    [index, height, vector n]
    and
    [[index_i, index_j], height, vector n]
    respectively.
    """
    
    points, edges = graph
    
    events_by_height = group_events_by_height(points, edges)
    
    # Mapping from index to height for points
    height_of_point = {i: h for i, h, n in points}

    # Union-Find Data Structures
    parent = {}
    rank = {}
    earliest_height = {}
    includes_previous = {}

    # Lists to record events
    new_components = []  # Records when a point creates a new connected component
    merges = []          # Records when connected components are merged
    
    
    # Union-Find helper functions
    def find(u):
        if parent[u] != u:
            parent[u] = find(parent[u])  # Path compression
        return parent[u]

    def union(u, v, current_height):
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            # Union by rank
            if rank[root_u] < rank[root_v]:
                parent[root_u] = root_v
                root = root_v
            else:
                parent[root_v] = root_u
                root = root_u
                if rank[root_u] == rank[root_v]:
                    rank[root_u] += 1
            # Update earliest_height and includes_previous
            earliest_height[root] = min(earliest_height[root_u], earliest_height[root_v])
            includes_previous[root] = (
                includes_previous[root_u] or
                includes_previous[root_v] or
                earliest_height[root] < current_height
            )
            return True
        return False
    
    # Processing events in order of height
    for h in sorted(events_by_height.keys()):
        current_events = events_by_height[h]
        # First process edges at height h
        for (i_j, h_e, n_e) in current_events['edges']:
            i, j = i_j
            # Ensure both points are in the union-find structure
            for idx in [i, j]:
                if idx not in parent:
                    parent[idx] = idx
                    rank[idx] = 0
                    earliest_height[idx] = height_of_point[idx]
                    includes_previous[idx] = earliest_height[idx] < h
            # Find roots of both points
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                # Check if either component includes previous heights
                includes_prev = (
                    includes_previous[root_i] or
                    includes_previous[root_j] or
                    earliest_height[root_i] < h or
                    earliest_height[root_j] < h
                )
                # Perform the union
                union(i, j, h)
                new_root = find(i)
                # If merging components from previous heights, record the merge
                if includes_previous[root_i] and includes_previous[root_j]:
                    # Determine the point causing the merge (the one with height h)
                    if height_of_point[i] == h:
                        point_causing_merge = i
                    elif height_of_point[j] == h:
                        point_causing_merge = j
                    else:
                        # If neither point has height h, default to one
                        point_causing_merge = i
                    merges.append({
                        'point_causing_merge': [point_causing_merge, height_of_point[point_causing_merge], points[i][2]],
                        'merged_components': {
                            'component_1_root': root_i,
                            'component_2_root': root_j
                        }
                    })
        # Now process points at height h
        recorded_roots = set()
        for i, h_i, n_i in current_events['points']:
            if i not in parent:
                parent[i] = i
                rank[i] = 0
                earliest_height[i] = h_i
                includes_previous[i] = False
            root_i = find(i)
            if earliest_height[root_i] == h and not includes_previous[root_i] and root_i not in recorded_roots:
                # Record the creation of a new connected component
                new_components.append({'point': [i, h_i, n_i]})
                recorded_roots.add(root_i)

    return new_components, merges



def process_graph(graph, direction):
    """
        The input is a graph containing points and edges and a direction.
        
        The output is a graph ordered by height, and by x,y,z. The normal vectors are replaced with the sign.
    """
    signed_graph = obtain_sign(graph, direction)
    processed_graph = order_graph(graph, direction)
    return processed_graph



# obtain_sign, sign, and order_graph are helper functions for process_graph. 

def obtain_sign(graph, direction):
    points, edges = graph
    signed_points = []
    signed_edges = []
    for point in points:
        signed_points.append([point[0],point[1],sign(point[2], direction)])
    for edge in edges:
        signed_edges.append([edge[0],edge[1],sign(edge[2], direction)])
    return [signed_points, signed_edges]

def sign(v_1,v_2):
    product = v_1[0] * v_2[0] +  v_1[1] * v_2[1] + v_1[2] * v_2[2]
    sign = 0
    if product > 0:
        sign = 1
    elif product < 0:
        sign = -1
    return sign

def order_graph(graph, direction):
    """
        The input is a graph containing points and edges and a direction.
        
        The output is a graph ordered by height, and by x,y,z.
    """
    ordered_graph = graph
    return ordered_graph
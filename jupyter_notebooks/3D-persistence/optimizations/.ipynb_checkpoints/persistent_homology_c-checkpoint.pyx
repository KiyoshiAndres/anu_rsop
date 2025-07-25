# persistent_homology_c.pyx   —  CYTHONIZED (22 Jul 2025)
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

from typing import List, Dict
import logging
from pathlib import Path

import numpy as np
cimport numpy as cnp

ctypedef cnp.int64_t INT_t
ctypedef cnp.double_t DTYPE_t

# ──────────────────────────────────────────────────────────────────────
#  0-dimensional persistence class
# ──────────────────────────────────────────────────────────────────────
class BettiZero:
    """Persistence for connected components (β₀)."""

    # ---------- disjoint-set ----------
    class UnionFind:
        def __init__(self, vertices: list[dict]):
            self.parent = list(range(len(vertices)))
            self.rank   = list(range(len(vertices)))

        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

        def union(self, x, y):
            rootX = self.find(x)
            rootY = self.find(y)
            rankX = self.rank[rootX]
            rankY = self.rank[rootY]
            if rootX == rootY:
                return
            if rankX < rankY:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                if rankX == rankY:
                    self.rank[rootX] += 1

    # ---------- life-cycle ----------
    def __init__(self, direction, vertices, edges):
        self.direction = direction
        self.vertices  = vertices
        self.edges     = edges

        # build filtration once
        self.filtration, self.new_to_original = make_filtration(
            vertices, edges, direction
        )

    def horizontal_step(self, edges, births, deaths):
        """
        When an edge appears with both endpoints already born, it kills the newest component.
        """
        uf = BettiZero.UnionFind(self.vertices)
        for e in edges:
            x, y = e['vertices']
            if uf.find(x) != uf.find(y):
                b = max(self.vertices[x]['birth'], self.vertices[y]['birth'])
                deaths.append((b, e['height']))
                uf.union(x, y)
        return deaths

    @staticmethod
    def vertical_step(edges, components, mergers, uf):
        """
        When two components merge, the one born later (larger new_index) dies.
        """
        for e in edges:
            x, y = e['vertices']
            root_before = (uf.find(x), uf.find(y))
            uf.union(x, y)
            if root_before[0] == root_before[1]:
                continue
            # The younger component dies
            younger = max(root_before)
            older   = min(root_before)
            mergers.append((components[younger], e['height']))
            components[younger] = components[older]
        return mergers

    def compute(self):
        """
        Run the full persistence algorithm, returning lists of births and deaths.
        """
        births, deaths = [], []
        # Step 1: 0D births at point appearances
        for h, ev in self.filtration.items():
            for p in ev['points']:
                births.append((h, h))
        # Step 2: horizontal edges create immediate deaths
        for h, ev in self.filtration.items():
            deaths = self.horizontal_step(ev['horizontal_edges'], births, deaths)
        # Step 3: vertical edges merge components
        uf = BettiZero.UnionFind(self.vertices)
        components = {i: i for i in range(len(self.vertices))}
        mergers = []
        for h in sorted(self.filtration):
            mergers = BettiZero.vertical_step(
                self.filtration[h]['vertical_edges'], components, mergers, uf
            )
        return births, deaths + mergers


# ──────────────────────────────────────────────────────────────────────
#  Filtration construction and event grouping
# ──────────────────────────────────────────────────────────────────────
def make_filtration(vertices, edges, direction):
    points = [[vertex, [0, 0, 0]] for vertex in vertices]
    pre_edges = edges
    pre_formatted_edges = []
    for edge in pre_edges:
        x, y = edge
        pre_formatted_edges.append([[x, y], [1, 1, 1]])
    pre_vertices = append_height_vertices(direction, points)
    vertices = format_vertices(pre_vertices)
    edges    = format_edges(vertices, pre_formatted_edges)
    pre_graph = process_graph(vertices, edges, direction)
    graph  = pre_graph['signed_graph']
    original_to_new = pre_graph['index_translation']
    new_to_original = {v: k for k, v in original_to_new.items()}
    filtration = group_events_by_height(graph[0], graph[1])
    return filtration, new_to_original


def group_events_by_height(points, edges):
    events_by_height = {}
    for point in points:
        h = point['height']
        events_by_height.setdefault(h, {
            'points': [], 'horizontal_edges': [], 'vertical_edges': []
        })
        events_by_height[h]['points'].append(point)

    for edge in edges:
        h_low, h_high = min(edge['height']), max(edge['height'])
        if h_low == h_high:
            events_by_height[h_high]['horizontal_edges'].append(edge)
        else:
            events_by_height[h_high]['vertical_edges'].append(edge)
    return events_by_height


def process_graph(vertices, edges, direction):
    processed = order_graph(vertices, edges)
    graph = [processed['vertices'], processed['edges']]
    signed = obtain_sign(graph, direction)
    return {'signed_graph': signed, 'index_translation': processed['index_translation']}


def subdivide_edges(edges: list) -> list:
    """Partition edges for processing in two stages (if needed)."""
    result = []
    for e in edges:
        result.append(e)
    return result


def obtain_sign(graph, direction: list) -> list:
    points, edges = graph
    signed_pts = []
    signed_edg = []
    for p in points:
        p['sign'] = sign(p['normal'], direction)
        del p['normal']
        signed_pts.append(p)
    for e in edges:
        signed_edg.append({
            'vertices': e['vertices'],
            'height'  : e['height'],
            'sign'    : sign(e['n'], direction)
        })
    return [signed_pts, signed_edg]


def order_graph(vertices, edges):
    sorted_vertices = sorted(
        vertices,
        key=lambda v: (v['height'],
                       v['coordinates'][0],
                       v['coordinates'][1],
                       v['coordinates'][2])
    )
    original_to_new = {
        v['original_index']: i for i, v in enumerate(sorted_vertices)
    }
    for v in sorted_vertices:
        v['new_index'] = original_to_new[v['original_index']]

    for e in edges:
        new_idx = [original_to_new[i] for i in e['vertices']]
        e['vertices'] = sorted(new_idx)

    sorted_edges = sorted(
        edges,
        key=lambda e: (max(e['height']), min(e['vertices']))
    )

    return {
        'vertices'         : sorted_vertices,
        'edges'            : [{'vertices': e['vertices'],
                               'height'  : e['height'],
                               'n'       : e['n']} for e in sorted_edges],
        'index_translation': original_to_new
    }


def sign(v_1, v_2):
    product = v_1[0]*v_2[0] + v_1[1]*v_2[1] + v_1[2]*v_2[2]
    if product > 0:
        return 1
    elif product < 0:
        return -1
    else:
        return 0


def height_of_vertex(direction, point):
    """Fast dot-product; >4× faster than Python loop."""
    return float(np.dot(direction, point))


def append_height_vertices(direction, vertices):
    pts = np.asarray([v[0] for v in vertices])
    heights = pts @ np.asarray(direction)
    return [[p.tolist(), h, n] for p, h, n in zip(
        pts, heights, (v[1] for v in vertices)
    )]


def format_vertices(vertices: list) -> list:
    new_vertices = []
    idx = 0
    for coord, height, normal in vertices:
        new_vertices.append({
            'coordinates'   : coord,
            'original_index': idx,
            'new_index'     : None,
            'height'        : height,
            'normal'        : normal
        })
        idx += 1
    return new_vertices


def format_edges(points: list, edges: list) -> list:
    formatted = []
    for edge in edges:
        lv, rv = edge[0]
        formatted.append({
            'vertices': [lv, rv],
            'height'  : [points[lv]['height'], points[rv]['height']],
            'n'       : edge[1]
        })
    return formatted


def rotate_points(points, theta, phi, alpha):
    # Create rotation matrices
    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    Ry = np.array([
        [ np.cos(phi), 0, np.sin(phi)],
        [ 0,           1, 0          ],
        [-np.sin(phi), 0, np.cos(phi)]
    ])
    Rx = np.array([
        [1, 0,            0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha),  np.cos(alpha)]
    ])
    R = Rz @ Ry @ Rx
    return points @ R.T


# ──────────────────────────────────────────────────────────────────────
#  Geometric helper routines (typed for speed)
# ──────────────────────────────────────────────────────────────────────

def generate_circle_points(int n, double radius=1.0):
    """
    Evenly spaced points on a circle in the XY-plane.
    Returns an (n×3) array of float64.
    """
    cdef cnp.ndarray[DTYPE_t, ndim=1] angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cdef cnp.ndarray[DTYPE_t, ndim=2] points = np.column_stack((
        radius * np.cos(angles),
        radius * np.sin(angles),
        np.zeros(n, dtype=np.float64)
    ))
    return points

def generate_sphere_points(int n, int rotations, double threshold):
    """
    Generate sphere by rotating the circle-points.
    Zeros out small coords and deduplicates.
    """
    # Use our typed circle-point function
    cdef cnp.ndarray[DTYPE_t, ndim=2] circle_points = generate_circle_points(n)
    cdef cnp.ndarray[DTYPE_t, ndim=1] angles = np.linspace(0, np.pi, rotations, endpoint=False)

    sphere_list = []
    cdef int i
    for i in range(rotations):
        phi = angles[i]
        # rotate_points is left untouched from your original file
        rotated = rotate_points(circle_points, 0, phi, 0)
        sphere_list.append(rotated)

    cdef cnp.ndarray[DTYPE_t, ndim=2] arr = np.vstack(sphere_list)
    arr[np.abs(arr) < threshold] = 0.0
    return np.unique(arr, axis=0)


"""
Reconstruct metric graph and extract features.

based on 2d implementation from:
    github: https://github.com/markolalovic/metric-graph-reconstruction
    author: @markolalovic
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import networkx as nx
import nibabel as nib
import os
import time
import datetime
import warnings
import argparse
import pathlib
import json
import multiprocessing
import vedo
import pandas as pd


class EmbeddedGraph:
    def __init__(self, nodes, edges, radius=[]):
        ''' Graph with points embedded in the plane.'''
        self.nodes = PointList(nodes)
        self.edges = [PointList(edge) for edge in edges]
        self.radius = radius

    def __str__(self):
        points = [str(point) for point in self.nodes.points]
        edges = [str(edge) for edge in self.edges]
        components = [str(cmpt_emb_G) for cmpt_emb_G in self.components.values()]

        return "nodes: {}\edges: {}\ncomponents: {}".format(
            str(points), str(edges), str(components))

    def to_dict(self):
        ''' Converts EmbeddedGraph to dict. '''
        points = [point.to_tuple() for point in self.nodes.points]
        edges = [edge.to_list() + [r] for edge, r in zip(self.edges, self.radius)]
        components = [cmpt_emb_G.to_list() for cmpt_emb_G in self.components.values()]

        embedded_graph = dict()
        embedded_graph["points"] = points
        embedded_graph["edges"] = edges
        embedded_graph["components"] = components

        return embedded_graph


    @property
    def n(self):
        ''' Number of nodes in EmbeddedGraph.'''
        return len(self.nodes.points)

    @property
    def m(self):
        ''' Number of edges in EmbeddedGraph.'''
        return len(self.edges)

    @property
    def k(self):
        ''' Number of connected components of EmbeddedGraph.'''
        return len(self.components)

    @property
    def components(self):
        ''' Computes connected components of EmbeddedGraph'''
        graph_G = graph(self)
        cmpts_G = graph_G.components

        cmpts_emb_G = {}
        point_of = {}
        for i in range(self.n):
            point_of[i] = self.nodes.points[i]

        for i, cmpt_G in cmpts_G.items():
            cmpts_emb_G[i] = PointList([point_of[j] for j in cmpt_G])

        return cmpts_emb_G


class Graph:
    def __init__(self, nodes, edges):
        ''' Graph represented with nodes and edges.'''
        if isinstance(nodes, list):
            self.nodes = nodes
        else:
            self.nodes = list(nodes)
        if isinstance(edges, list):
            self.edges = edges
        else:
            self.edges = list(edges)

    @property
    def n(self):
        ''' Number of nodes in Graph.'''
        return len(self.nodes)

    @property
    def m(self):
        ''' Number of edges in Graph.'''
        return len(self.edges)

    @property
    def k(self):
        ''' Number of connected components of Graph.'''
        return len(self.components)

    @property
    def components(self):
        ''' Computes connected components of Graph'''
        cmpts = {}
        k = 0
        unvisited = copy.copy(self.nodes)
        for v in self.nodes:
            if v in unvisited:
                comp_of_v = component(v, self.nodes, self.edges)
                # remove visited nodes in component from unvisited
                unvisited = list(set(unvisited) - set(comp_of_v))
                cmpts[k] = comp_of_v
                k += 1

        return cmpts

    def __str__(self):
        return "nodes: {}\nedges: {}".format(str(self.nodes), str(self.edges))

    def draw(self):
        graph_G = nx.Graph()
        graph_G.add_nodes_from(self.nodes)
        graph_G.add_edges_from(self.edges)

        pos = nx.spring_layout(graph_G)
        nx.draw(graph_G, pos, font_size=10,
                node_color='red', with_labels=True)
        plt.show()


def nhbs(v, graph_G):
    N = []
    for edge in graph_G.edges:
        u1, u2 = edge
        if u1 == v:
            N.append(u2)
        elif u2 == v:
            N.append(u1)
    return N


def component(v, nodes, edges):
    ''' Wrapper of comp.'''
    G = Graph(nodes, edges)
    return comp(v, G, [v])  # T=[v] at the start


def comp(v, graph_G, T):
    N = list(set(nhbs(v, graph_G)) - set(T))
    if N == []:
        return [v]
    else:
        T += N  # expand the tree
        for n in N:
            T += comp(n, graph_G, T)  # expand the tree (BFS)
    return list(set(T))


def graph(emb_G):
    ''' Translate from EmbeddedGraph to Graph.'''

    point_of = {}
    for i in range(emb_G.n):
        point_of[i] = emb_G.nodes.points[i]

    number_of = {}
    for i in range(emb_G.n):
        number_of[emb_G.nodes.points[i]] = i

    nodes = list(point_of.keys())
    edges = []
    for i in range(emb_G.n):
        for j in range(i + 1, emb_G.n):
            # test if there is an edge between Points v1 and v2
            v1 = emb_G.nodes.points[i]
            v2 = emb_G.nodes.points[j]

            for edge in emb_G.edges:
                u1 = edge.points[0]
                u2 = edge.points[1]
                if v1.equal(u1) and v2.equal(u2) or \
                        v1.equal(u2) and v2.equal(u1):
                    edges.append((number_of[v1], number_of[v2]))

    return Graph(nodes, edges)


class Point:
    ''' Class Point for storing coordinates and label of a point.

    Args:
        x::float
            The x coordinate of the point.
        y::float
            The y coordinate of the point.
        label::str
            Should be: 'E' for edge point and 'V' for vertex point.
    '''

    def __init__(self, x=0, y=0, z=0, radius=0, label=''):
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius
        if label not in ('E', 'V', ''):
            raise ValueError("Label must be 'E' or 'V'")
        self.label = label

    def __str__(self):
        return "({}, {}, {})".format(self.x, self.y, self.z, self.label)

    def to_tuple(self):
        '''Converts Point to tuple.'''
        return self.x, self.y, self.z

    def equal(self, p):
        return (self.x == p.x) and (self.y == p.y) and (self.z == p.z)


class PointList:
    def __init__(self, points):
        ''' PointList Class to hold a list of Point objects.'''
        if points == [] or isinstance(points[0], Point):
            self.points = points
        else:
            raise ValueError("Args must be a list of Points.")

    @property
    def vertex_points(self):
        vertex_points = []
        for point in self.points:
            if point.label == 'V':
                vertex_points.append(point)

        return vertex_points

    @property
    def edge_points(self):
        edge_points = []
        for point in self.points:
            if point.label == 'E':
                edge_points.append(point)

        return edge_points

    @property
    def center(self):
        ''' Center of mass of the point cloud.'''
        x = np.mean(np.array([point.x for point in self.points]))
        y = np.mean(np.array([point.y for point in self.points]))
        z = np.mean(np.array([point.z for point in self.points]))
        radius = np.mean(np.array([point.radius for point in self.points]))

        return Point(x, y, z, radius)

    def __str__(self):
        return '[' + ','.join(['{!s}'.format(p) for p in self.points]) + ']'

    def to_list(self):
        '''Converts PointList to list.'''
        return [p.to_tuple() for p in self.points]

    def __len__(self):
        return len(self.points)

    def contains(self, p):
        for pt in self.points:
            if pt.x == p.x and pt.y == p.y and pt.z == p.z:
                return True
        return False

    def append(self, p):
        self.points.append(p)

    def difference(self, pl):
        difference = PointList([])
        for pt in self.points:
            if not pl.contains(pt):
                difference.append(pt)
        return difference

    def distance(self, point_list):
        ''' Computes minimum distance from self to another point list.'''
        distances = []
        for p1 in self.points:
            for p2 in point_list.points:
                distances.append(distance(p1, p2))

        return np.min(np.array(distances))

    def avg_radius(self):
        radius = []
        for p in self.points:
            radius.append(p.radius)
        return np.array(radius).mean()


class Canvas:
    """ Class Canvas on which we draw the graphics."""

    def __init__(self, title, xlabel='X', ylabel='Y', zlabel='Z',
                 p1=Point(-2, -2, -2, 0), p2=Point(100, 100, 100, 0)):
        self.fig = plt.figure()
        self.fig.set_size_inches(50, 50)
        self.ax = self.fig.add_subplot(111, aspect='auto', projection='3d')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        self.ax.set_zlabel('Z')
        plt.xticks(list(range(p1.x, p2.x))[0::20])
        plt.yticks(list(range(p1.y, p2.y))[0::20])
        self.ax.set_zticks(list(range(p1.z, p2.z))[0::20])
        self.ax.grid(True)
        self.ax.set_xlim([p1.x, p2.x])
        self.ax.set_ylim([p1.y, p2.y])
        self.ax.set_zlim([p1.z, p2.z])

    def show(self):
        """ Show the canvas, displaying any graphics drawn on it."""
        plt.show()
        plt.clf()
        plt.cla()
        plt.close()

    def save(self, name):
        self.fig.savefig(name)


def draw_point(canvas, pt, radius=0.25, color='blue', **kwargs):
    ''' Draws a point.'''
    z_points = pt.z
    x_points = pt.x
    y_points = pt.y
    canvas.ax.scatter3D(x_points, y_points, z_points, c=color);


def draw_points(canvas, points):
    for point in points:
        if point.label == 'V':
            color = 'red'
        elif point.label == 'E':
            color = 'blue'
        else:
            color = 'green'
        draw_point(canvas, point, color=color)


def distance(p1, p2):
    ''' Euclidean distance between p1, p2.'''
    d = (
            ((p1.x - p2.x) ** 2) +
            ((p1.y - p2.y) ** 2) +
            ((p1.z - p2.z) ** 2)
        ) ** 0.5
    return d


def get_shell_points(points, center, r, delta):
    ''' Returns a list of points between r and r + delta around the center
    point.'''
    shell_points = []
    for point in points:
        d = distance(center, point)
        if d >= r and d <= r + delta:
            shell_points.append(point)

    return shell_points


def get_ball_points(points, center, r):
    ball_points = []
    for point in points:
        d = distance(center, point)
        if d < r:
            ball_points.append(point)

    return ball_points


def rips_vietoris_graph(delta, points):
    ''' Constructs the Rips-Vietoris graph of parameter delta whose nodes
    are points of the shell.'''
    n = len(points)
    nodes = []
    edges = []
    for i in range(n):
        p1 = points[i]
        nodes.append(p1)
        for j in range(i, n):
            p2 = points[j]
            if not p1.equal(p2) and distance(p1, p2) < delta:
                edges.append([p1, p2])

    return EmbeddedGraph(nodes, edges)


def reconstruct(point_list, delta=3, r=2, p11=1.5, show=False):
    ''' Implementation of Aanjaneya's metric graph reconstruction algorithm.'''
    # label the points as edge or vertex points
    for center in point_list.points:
        shell_points = get_shell_points(point_list.points, center, r, delta)
        rips_embedded = rips_vietoris_graph(delta, shell_points)

        if rips_embedded.k == 2:
            center.label = 'E'
        else:
            center.label = 'V'
    if show:
        canvas = Canvas('After labeling')
        draw_points(canvas, point_list.points)

    # re-label all the points withing distance p11 from vertex points as vertices
    for center in point_list.vertex_points:
        ball_points = get_ball_points(point_list.edge_points, center, p11)
        for ball_point in ball_points:
            ball_point.label = 'V'
    if show:
        canvas = Canvas('After re-labeling')
        draw_points(canvas, point_list.points)

    # reconstruct the graph structure
    # compute the connected components of Rips-Vietoris graphs:
    # R_delta(vertex_points), R_delta(edge_points)
    rips_V = rips_vietoris_graph(delta, point_list.vertex_points)
    rips_E = rips_vietoris_graph(delta, point_list.edge_points)
    cmpts_V = rips_V.components
    cmpts_E = rips_E.components

    # DEBUG:
    nodes_emb_E = []
    for i, cmpt_E in cmpts_E.items():
        nodes_emb_E.append(cmpt_E.center)
    emb_E = EmbeddedGraph(nodes_emb_E, [])

    nodes_emb_G = []
    for i, cmpt_V in cmpts_V.items():
        nodes_emb_G.append(cmpt_V.center)

    n = len(nodes_emb_G)
    edges_emb_G = []
    radius = []
    for i in range(n):
        for j in range(i + 1, n):
            for cmpt_E in cmpts_E.values():
                if cmpts_V[i].distance(cmpt_E) < delta and \
                        cmpts_V[j].distance(cmpt_E) < delta:
                    edges_emb_G.append([nodes_emb_G[i], nodes_emb_G[j]])
                    radius.append(cmpt_E.avg_radius())

    emb_G = EmbeddedGraph(nodes_emb_G, edges_emb_G, radius)
    if show:
        canvas = Canvas('Result')
        draw_points(canvas, point_list.points)
        draw_graph(canvas, emb_G, color='red')
        draw_graph(canvas, emb_E, color='black')
        print(emb_E)

    return emb_G


def draw_labeling(point_list, delta=3, r=2, p11=1.5, step=0, savefig=False):
    ''' Draw the labeling step of the algorithm.'''

    canvas = Canvas('Labeling points as edge or vertex points')
    draw_points(canvas, point_list.points)

    if step == 0:
        step = int(np.floor(len(point_list.points) / 4)) - 2
    center = point_list.points[step]

    # draw_ball(canvas, center, r, 'black')
    # draw_ball(canvas, center, r + delta, color='black')

    shell_points = get_shell_points(point_list.points, center, r, delta)
    rips_embedded = rips_vietoris_graph(delta, shell_points)

    draw_graph(canvas, rips_embedded, color='red')  # , savefig=True)

    if savefig:
        canvas.save("labeling.svg")

    plt.show()


def draw_re_labeling(point_list, delta=3, r=2, p11=1.5, savefig=False):
    # label points as edge or vertex
    for center in point_list.points:
        shell_points = get_shell_points(point_list.points, center, r, delta)
        rips_embedded = rips_vietoris_graph(delta, shell_points)

        if rips_embedded.k == 2:
            center.label = 'E'
        else:
            center.label = 'V'

    canvas = Canvas('Re-labeling points as vertex points')
    draw_points(canvas, point_list.points)

    i = int(np.floor(len(point_list.points) / 4)) - 2
    center = point_list.points[i]

    # draw_ball(canvas, center, radius=p11, color='black')

    # ball_points = get_ball_points(point_list.edge_points, center, p11)
    # for ball_point in ball_points:
    #    draw_point(canvas, ball_point, color='green')

    if savefig:
        canvas.save(f"re_labeling_{delta}.png")

    plt.show()


def draw_graph(canvas, emb_G, color='black', savefig=False):
    for pt in emb_G.nodes.points:
        draw_point(canvas, pt, color=color)

    for edge in emb_G.edges:
        draw_edge(canvas, edge.points[0], edge.points[1], color=color)

    if savefig:
        canvas.save("graph.svg")

    plt.show()


def draw_ball(canvas, pt, radius=5, color='blue', **kwargs):
    """ Draws a ball."""
    z_points = pt.z
    x_points = pt.x
    y_points = pt.y
    canvas.ax.scatter3D(x_points, y_points, z_points, c=color)


def draw_edge(canvas, p1, p2, color='blue', **kwargs):
    z_line = [p1.z, p2.z]
    x_line = [p1.x, p2.x]
    y_line = [p1.y, p2.y]
    canvas.ax.plot3D(x_line, y_line, z_line, color)


def run_reconstruction(filenames, out_vtk, out_json):
    """
    Runs the metric graph reconstruction for filenames and saves the result as json.
    Furthermore its visualization is saved as vtk file.
    """
    for fn_path in filenames:
        print(f"Reconstructing {fn_path.name} ...")
        ###################################################################
        # PREPROCESS FILENAME
        ###################################################################
        idx_1 = fn_path.name.find('_')
        idx_2 = fn_path.name.find('_', idx_1 + 1)
        DATETIME = fn_path.name[idx_1:idx_2 + 1]

        # extract the 3 digit id + measurement string eg PAT001_RL01
        idxID = fn_path.name.find('PAT')

        if idxID == -1:
            idxID = fn_path.name.find('VOL')

        if idxID is not -1:
            ID = fn_path.name[idxID:idxID + 11]
        else:
            # ID is different, extract string between Second "_" and third "_"
            # usually it is 6 characters long
            idx_3 = fn_path.name.find('_', idx_2 + 1)
            ID = fn_path.name[idx_2 + 1:idx_3]
        ###################################################################
        # RECONSTRUCTION
        ###################################################################
        # load volume and create PointList object
        vol = nib.load(str(fn_path)).get_data()
        vol = np.flip(vol, 0)
        points = []
        x = vol.shape[2]
        y = vol.shape[1]
        z = vol.shape[0]
        for x_idx in range(x):
            for y_idx in range(y):
                for z_idx in range(z):
                    radius = vol[z_idx, y_idx, x_idx]
                    if radius > 0.:
                        points.append(Point(x_idx, y_idx, z_idx, round(radius, 3)))
        # inputs to the algorithm
        point_list = PointList(points)
        delta, r, p11 = 2, 1.5, 0.9
        # reconstruct metric graph
        reconstructed = reconstruct(point_list, delta, r, p11)
        ###################################################################
        # SAVE RESULT
        ###################################################################
        graph = reconstructed.to_dict()
        graph["shape"] = [vol.shape[2], vol.shape[1], vol.shape[0]]
        # save as vtk file
        edges = [edge[:2] for edge in graph["edges"]]
        edges = vedo.shapes.Lines(edges)
        edges_radius = [edge[2] for edge in graph["edges"]]
        edges.cellColors(edges_radius, cmap='jet').addScalarBar3D(c='k')
        nodes = vedo.pointcloud.Points(graph["points"])
        out_nodes = f"{out_vtk}/R{DATETIME}{ID}_nodes.vtk"
        out_edges = f"{out_vtk}/R{DATETIME}{ID}_edges.vtk"
        vedo.io.write(nodes, out_nodes)
        vedo.io.write(edges, out_edges)
        # save nodes, edges, components and shape as json
        with open(f"{out_json}/R{DATETIME}{ID}_metric_graph.json", "w") as f:
            json.dump(graph, f)
        print(f"Done reconstructing {fn_path.name} !")


def compute_distance(p1, p2):
    ''' Euclidean distance between tuples p1 and p2.'''
    d = (
            ((p1[0] - p2[0]) ** 2) +
            ((p1[1] - p2[1]) ** 2) +
            ((p1[2] - p2[2]) ** 2)
        ) ** 0.5
    return d


def compute_distance_mc(p1, p2):
    ''' Euclidean distance between tuples p1 and p2 in micrometer (given a RSOM pixel space).'''
    d = (
            (((p1[0] - p2[0]) * 12) ** 2) +
            (((p1[1] - p2[1]) * 12) ** 2) +
            (((p1[2] - p2[2]) * 3) ** 2)
        ) ** 0.5
    return d


def get_features(path_to_json_dir, out_vtk, h_params=None):
    """
    Extract features from metric graph json files.
    """

    if h_params is None:
        h_params = dict()
        h_params["min_component_length"] = 10
        h_params["min_total_length"] = 1000
        h_params["min_end_branch_length"] = 20

    noisy_samples = []
    path = pathlib.Path(path_to_json_dir)
    filenames = set(path.glob("*graph.json"))

    main_features = [
                     "total_vessel_length",
                     "small_vessel_length",
                     "large_vessel_length",
                     "#vessel_bifurcations"
                     ]

    additional_features = [
                            "avg_radius",
                            "#components",
                            "length_per_component",
                            "avg_path_length",
                            "density",
                            "degree_assortativity_coefficient",
                            "#cycles"
                           ]

    columns = ["filename"] + main_features + additional_features

    rows = []

    for f in filenames:
        fn_path = pathlib.Path(f)
        ###################################################################
        # PREPROCESS FILENAME
        ###################################################################
        idx_1 = fn_path.name.find('_')
        idx_2 = fn_path.name.find('_', idx_1 + 1)
        DATETIME = fn_path.name[idx_1:idx_2 + 1]

        # extract the 3 digit id + measurement string eg PAT001_RL01
        idxID = fn_path.name.find('PAT')

        if idxID == -1:
            idxID = fn_path.name.find('VOL')

        if idxID is not -1:
            ID = fn_path.name[idxID:idxID + 11]
        else:
            # ID is different, extract string between Second "_" and third "_"
            # usually it is 6 characters long
            idx_3 = fn_path.name.find('_', idx_2 + 1)
            ID = fn_path.name[idx_2 + 1:idx_3]

        ###################################################################
        # PROCESS JSON
        ###################################################################
        with open(f) as f_in:
            graph = json.load(f_in)

        nodes = [tuple(n) for n in graph["points"]]
        edges = [[tuple(e) for e in edge[:2]] + [edge[2]] for edge in graph["edges"]]

        if len(nodes) == 0:
            noisy_samples.append("_".join(f.name.split("_")[:3]))
            print(f.name, "removed due to no nodes before clean")
            continue

        G = nx.Graph()
        G.add_weighted_edges_from(edges, weight='radius')
        G.add_nodes_from(nodes)

        ###################################################################
        # PREPROCESSING
        ###################################################################
        # remove samples with a total length smaller than h_params["min_total_length"]
        total_length = 0
        for edge in G.edges:
            total_length += compute_distance(edge[0], edge[1])

        if total_length < h_params["min_total_length"]:
            noisy_samples.append("_".join(f.name.split("_")[:3]))
            print(f.name, "removed due to minimum total length")
            continue

        # remove components smaller than h_params["min_component_length"]
        G_clean = nx.Graph()
        for c in nx.connected_components(G):
            distance = 0
            g = G.subgraph(c)
            for edge in g.edges:
                distance += compute_distance(edge[0], edge[1])
            if distance > h_params["min_component_length"]:
                G_clean = nx.compose(G_clean, g)

        # remove end-branches smaller than h_params["min_end_branch_length"]: (for one iteration)
        edges_to_remove = []
        for e in G_clean.edges():
            if compute_distance(e[0], e[1]) < h_params["min_end_branch_length"]:
                if G_clean.degree(e[0]) == 1 or G_clean.degree(e[1]) == 1:
                    edges_to_remove.append(e)
        G_clean.remove_edges_from(edges_to_remove)

        # remove isolated nodes eventually caused by previous edge removement
        G_clean.remove_nodes_from(list(nx.isolates(G_clean)))
        if G_clean.number_of_nodes() == 0:
            noisy_samples.append("_".join(f.name.split("_")[:3]))
            print(f.name, "removed due to no nodes after clean")
            continue

        # save clean metric graph as vtk
        edges = G_clean.edges()
        edges = vedo.shapes.Lines(edges)
        edges_radius = [edge[2]["radius"] for edge in G_clean.edges(data=True)]
        edges.cellColors(edges_radius, cmap='jet').addScalarBar3D(c='k')
        nodes = vedo.pointcloud.Points([list(n) for n in G_clean.nodes()])
        out_nodes = f"{out_vtk}/R{DATETIME}{ID}_nodes_clean.vtk"
        out_edges = f"{out_vtk}/R{DATETIME}{ID}_edges_clean.vtk"
        vedo.io.write(nodes, out_nodes)
        vedo.io.write(edges, out_edges)

        ###################################################################
        # FEATURE EXTRACTION
        ###################################################################
        # AVERAGE PATH LENGTH IN MM
        path_lengths = []
        for v in G_clean.nodes():
            spl = dict(nx.single_source_shortest_path_length(G_clean, v))
            for p in spl:
                path_lengths.append(spl[p])

        avg_path_length = (sum(path_lengths) / len(path_lengths)) / 1000
        # DENSITY
        density = nx.density(G_clean)
        # DEGREE ASSORTATIVITY COEFFICIENT
        degree_assortativity_coefficient = nx.degree_assortativity_coefficient(G_clean)
        if degree_assortativity_coefficient is None:
            degree_assortativity_coefficient = 0
        # NUMBER OF CYCLES
        num_cycles = len(nx.cycle_basis(G_clean))
        # NUMBER OF BIFURCATION POINTS
        num_vessel_bifurcations = len([val for (node, val) in G_clean.degree() if val > 2])
        # NUMBER OF COMPONENTS AND AVG LENGTH PER COMPONENT IN MM
        num_components = nx.number_connected_components(G_clean)
        len_components = []
        for c in nx.connected_components(G_clean):
            distance = 0
            g = G_clean.subgraph(c)
            for edge in g.edges:
                distance += compute_distance_mc(edge[0], edge[1])
            len_components.append(distance)
        len_components = np.array(len_components)
        length_per_component = len_components.mean() / 1000
        # TOTAL LENGTH OF METRIC GRAPH IN MM
        total_vessel_length = len_components.sum() / 1000
        # AVERAGE RADIUS
        avg_radius = np.array([e[2]["radius"] for e in G_clean.edges(data=True)]).mean()
        # NUMBER OF SMALL AND LARGE VESSELS in MM
        small_vessel_length = 0
        large_vessel_length = 0
        for e in G_clean.edges(data=True):
            if e[2]["radius"] <= 2.5:
                small_vessel_length += compute_distance_mc(e[0], e[1])
            else:
                large_vessel_length += compute_distance_mc(e[0], e[1])
        small_vessel_length /= 1000
        large_vessel_length /= 1000

        rows.append(
            ["_".join(f.name.split("_")[:3]),
             total_vessel_length,
             small_vessel_length,
             large_vessel_length,
             num_vessel_bifurcations,
             avg_radius,
             num_components,
             length_per_component,
             avg_path_length,
             density,
             degree_assortativity_coefficient,
             num_cycles,
             ])

    print(len(noisy_samples), "samples were removed due to noise.")

    # construct pandas dataframe
    df = pd.DataFrame(rows, columns=columns)
    df = df.set_index('filename')

    return df

# ported from https://github.com/BenjaminFiltjens/MS-GCN/blob/f20deb80b8d11bf8f64e8a6e0622e9dea6b4acfd/models/net_utils/graph.py

import numpy as np

# obtained from ./analysis/graph.ipynb
link_beat = [(0, 1), (0, 63), (0, 69), (1, 2), (2, 3), (3, 4), (4, 5), (4, 9), (4, 36), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (12, 13), (12, 17), (12, 27), (13, 14), (14, 15), (15, 16), (17, 18), (17, 22), (18, 19), (19, 20), (20, 21), (22, 23), (23, 24), (24, 25), (25, 26), (27, 28), (27, 32), (28, 29), (29, 30), (30, 31), (32, 33), (33, 34), (34, 35), (36, 37), (37, 38), (38, 39), (39, 40), (39, 44), (39, 54), (40, 41), (41, 42), (42, 43), (44, 45), (44, 49), (45, 46), (46, 47), (47, 48), (49, 50), (50, 51), (51, 52), (52, 53), (54, 55), (54, 59), (55, 56), (56, 57), (57, 58), (59, 60), (60, 61), (61, 62), (63, 64), (64, 65), (65, 66), (66, 67), (67, 68), (69, 70), (70, 71), (71, 72), (72, 73), (73, 74)]


class Graph:
    """ The Graph to model the skeletons extracted by the openpose
    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).
        layout (string): must be one of the follow candidates
        - tp-vicon: 9 joints (marker-based mocap lower-body)
        - hugadb: 6 joints (IMU-based mocap lower-body)
        - lara: 19 joints (marker-based mocap full-body)
        - pku-mmd: x joints (markerless-based mocap full-body)
        - tug dataset uses lara graph: 19 joints (marker-based mocap full-body)
        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points
    """

    def __init__(self,
                 layout='tp-vicon',
                 strategy='spatial',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'tp-vicon':
            self.num_node = 9
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(1, 0), (2, 1), (3, 2), (4, 3), (5, 0), (6, 5), (7, 6), (8, 7)]
            self.edge = self_link + neighbor_link
            self.center = 0
        elif layout == 'hugadb':
            self.num_node = 6
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(1, 0), (2, 1), (3, 0), (4, 3), (5, 0)]
            self.edge = self_link + neighbor_link
            self.center = 0
        elif layout == 'lara':
            self.num_node = 19
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(1, 0), (2, 1), (3, 2), (4, 3), (5, 0), (6, 5), (7, 6), (8, 7), (9, 0), (10, 9), (11, 9), (12,10), (13,12), (14,13), (15,9), (16,15), (17,16), (18,17)]
            self.edge = self_link + neighbor_link
            self.center = 0
        elif layout == 'pku-mmd':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(12, 0), (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17), (19, 18), (1, 0), (20, 1), (2, 20), (3, 2), (4,20), (5,4), (6,5), (7,6), (21,7), (22,6), (8,20), (9,8), (10, 9), (11,10), (24,10), (23,11)]
            self.edge = self_link + neighbor_link
            self.center = 0
        elif layout == 'beat':
            self.num_node = 75
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = link_beat
            self.edge = self_link + neighbor_link
            self.center = 0
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_undigraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD
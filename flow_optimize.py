import numpy as np
from copy import deepcopy
from random import choice
#from projecting_linear import LinearWorker
#from projecting_sparse import SparseWorker
from projection import ProjectionWorker
from projection import ProjectionSurface
#from projecting_sparse import theta_project_sparse
from rb_tree import RBTree, RBTreeNode

def initial_flows(graph, corresp_matrix, path_dict):
    array = dict()
    for pair,paths in path_dict.items():
        amount = corresp_matrix[pair]
        el = np.zeros(shape=len(paths))
        el[0] = amount
        if len(paths) == 0:
            continue
        p = paths[0]
        for i in range(len(p)-1):
            graph.get_edge(p[i],p[i+1]).add_users(amount)
        array[pair] = el
    return array


def flow_optimize(graph, corresp_matrix, path_dict, worker: ProjectionWorker):
    iter = 0
    print('Iter',iter)
    x = initial_flows(graph, corresp_matrix, path_dict)
    graph.log_amount()
    y = x.copy()
    worker_x_dict = dict()
    worker_y_dict = dict()
    for pair, x_i in x.items():
        worker_x = worker(vector_struct=x_i,z=corresp_matrix[pair],project_type=ProjectionSurface.SIMPLEX)
        worker_x_dict[pair] = worker_x
    for pair, y_i in y.items():
        worker_y = worker(vector_struct=y_i,z=corresp_matrix[pair],project_type=ProjectionSurface.SIMPLEX)
        worker_y_dict[pair] = worker_y
    graph_x = graph
    graph_y = deepcopy(graph)
    iter = 1

    while iter <= 1000:
        pair, paths = choice(list(path_dict.items()))
        y_i = y[pair]
        x_i = x[pair]
        worker_y = worker_y_dict[pair]
        worker_x = worker_x_dict[pair]
        y_i_new = x_i - 1.0/iter * np.array([graph_x.get_cost(p) for p in paths])
        worker_y.update(y_i_new)
        y_i_new = worker_y.project()
        worker_y.update(y_i_new)

        diff_y = y_i_new - y_i
        graph_y.change_amount(paths, diff_y)

        x_i_new = x_i - 1.0 / iter * np.array([graph_y.get_cost(p) for p in paths])
        worker_x.update(x_i_new)
        x_i_new = worker_x.project()
        worker_x.update(x_i_new)

        diff_x = x_i_new - x_i
        graph_x.change_amount(paths, diff_x)

        #graph_x.log_amount()

        x[pair] = x_i_new
        y[pair] = y_i_new

        iter += 1
    print('Last iter')
    graph.log_amount()

import argparse
import time
import sys
import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tables
import mmap
import tempfile
import operator
import shutil
import multiprocessing
import plotly.plotly as py
import plotly.graph_objs as go
from multiprocessing import Pool
from functools import partial
from networkx.utils import not_implemented_for


parser = argparse.ArgumentParser(description="pagerank")
parser.add_argument('-f', '--file', default="web-Google.txt",
                    help="File location. If empty, Use web-Google.txt download from http://snap.stanford.edu/data/web-Google.html")
parser.add_argument('-m', '--matrix', dest="matrix",
                    action="store_true", help="Use tradition matrix mode instead of Graph mode,requires higher memory")
parser.add_argument('-p', '--probability', default = 0.85,
                    help="Probability for jumping to other random webpage. Solves dead-ends and spider traps")
parser.add_argument('-t', '--processes', default = 20,
                    help="Multi-processes for large matrix")                   
args = parser.parse_args()

def file_len():
    with open(args.file) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

#------------------------------------------Progress Bar------------------------------------------
# From: https://stackoverflow.com/questions/3160699/python-progress-bar/55003596
def update_progress(progress):
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

#------------------------------------------Graph Mode------------------------------------------
# Put all nodes and edges into graph 
def file_handle_graph(G,file_len):
    i = 0
    node_file = open(args.file, "r")
    print("Reading File into graph...\n")
    #if its comment line
    for line in node_file:
        line = line.rstrip()

        if line[0] == "#":
            continue

        line_arr = line.split("\t")
        add_node_to_graph(G, int(line_arr[0]))
        add_node_to_graph(G, int(line_arr[1]))
        add_edge_to_graph(G,int(line_arr[0]),int(line_arr[1]))
        update_progress(i/file_len)
        i += 1

    node_file.close()
    print("\nReading complete\n")

def creat_network_graph():
    G = nx.DiGraph()
    return G

def add_node_to_graph(G, nodeid):
    if G.has_node(nodeid):
        pass
    else:
       G.add_node(nodeid)

def add_edge_to_graph(G, fromid, toid):
    if G.has_edge(fromid,toid):
        pass
    else:
        G.add_edge(fromid, toid)

def pagerank(G, alpha=0.85, personalization=None,
             max_iter=2000, tol=1.0e-9, nstart=None, weight='weight',
             dangling=None):
    
    if len(G) == 0:
        return {}

    print("Creating a copy in stochastic form")
    W = nx.stochastic_graph(G, weight=weight)
    N = W.number_of_nodes()


    print("Choosing fixed starting vector")
    if nstart is None:
        x = dict.fromkeys(W, 1.0 / N)
    else:
        # Normalized nstart vector
        s = float(sum(nstart.values()))
        x = dict((k, v / s) for k, v in nstart.items())
    
    print("Assigning personalization vector")
    if personalization is None:
        # Assign uniform personalization vector if not given
        p = dict.fromkeys(W, 1.0 / N)
    else:
        s = float(sum(personalization.values()))
        p = dict((k, v / s) for k, v in personalization.items())

    if dangling is None:
        # Use personalization vector if dangling vector not specified
        dangling_weights = p
    else:
        s = float(sum(dangling.values()))
        dangling_weights = dict((k, v / s) for k, v in dangling.items())
    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]

    i = 0
    print("power iterating")
    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):  
        update_progress( i/max_iter *120)
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
        for n in x:
            # this matrix multiply looks odd because it is
            # doing a left multiply x^T=xlast^T*W
            for nbr in W[n]:
                x[nbr] += alpha * xlast[n] * W[n][nbr][weight]
            x[n] += danglesum * dangling_weights.get(n, 0) + (1.0 - alpha) * p.get(n, 0)
        # check convergence, l1 norm
        err = sum([abs(x[n] - xlast[n]) for n in x])
        if err < N * tol:
            return x
        i+=1
    raise nx.PowerIterationFailedConvergence(max_iter)

#------------------------------------------Matrix Mode------------------------------------------
# File handling

def file_handle_matrix(file_len):
    i = 0
    max_node_id = 0
    shutil.copyfile(args.file, '/dev/shm/temp.txt')
    node_file = open("/dev/shm/temp.txt", "r")
    print("Reading File...\n")
    #if its comment line
    for line in node_file:
        line = line.rstrip()
        if line[0] == "#":
            continue

        line_arr = line.split("\t")

        if ( int(line_arr[0])> max_node_id ):
            max_node_id = int(line_arr[0])
        if ( int(line_arr[1])> max_node_id ):
            max_node_id = int(line_arr[1])
        update_progress(i/file_len)
        i += 1
    node_file.close()
    print("\nReading complete\n")
    os.remove("/dev/shm/temp.txt")
    return max_node_id + 1


#how many unique outbound in toatl from a single node?
def node_outbound(nodeid):
    shutil.copyfile(args.file, '/dev/shm/temp.txt')
    node_file = open("/dev/shm/temp.txt", "r")
    #print("Counting outbound for node",nodeid)

    unique_tolist = []
    for line in node_file:
        line = line.rstrip()
        if line[0] == "#":
            continue
        line_arr = line.split("\t")
        if ( int(line_arr[0]) == nodeid ):
            if int(line_arr[1]) in unique_tolist:
                pass
            else:
                unique_tolist.append( int(line_arr[1]) )
    #print("Total outbound: ", len(unique_tolist))
    #print("Outbound: ",unique_tolist)
    os.remove("/dev/shm/temp.txt")
    return unique_tolist


# random browser factor 0.2
def fill_matrix(original_matrix,n):
    node_file = open(args.file, "r")
    print("Filiing Matrix\n")
    #if its comment line
    for y in range(0, n):
        unique_tolist = node_outbound(y)
        for x in range(0, n):
            #print("X: ",x)
            #print("Y: ",y)
            if ( (x != y) and ( x in unique_tolist) ):
                original_matrix[x][y] = 1 / len(unique_tolist)
            else:              
                original_matrix[x][y] = 0
    node_file.close()
    print("\nOriginal Matrix: \n", original_matrix)

def markov_matrix(original_matrix,n):
    pr = np.zeros((n, 1), dtype=float)
    for i in range(n):
        pr[i] = float(1) / n
    return (pr)

def pagerank_matrix(probability, original_matrix, v):
    while ((v == probability * np.dot(original_matrix, v) + (1 - probability) * v).all() == False):
        v = probability * np.dot(original_matrix, v) + (1 - probability) * v
        return v

def pagerank_top10(v,n):
    print_dict = {}
    for i in range(0, n):
        print_dict[i] = float(v[i])
    print("\nResult:")
    print(sorted(print_dict.items(), key=operator.itemgetter(1), reverse = True)[:10])

#------------------------------------------Large Matrix Mode------------------------------------------
def fill_matrix_by_column(n,result_dict,x):

    onerow_matrix = np.zeros((1,n), dtype = float)
    print("Filiing Large Matrix")
    #if its comment line
    for y in range(0, n):
        unique_tolist = node_outbound(y)
        if ( (x != y) and ( x in unique_tolist) ):
            try:
                onerow_matrix[0][y] = 1 / len(unique_tolist)
            except IndexError:
                return
        else:
            try:              
                onerow_matrix[0][y] = 0
            except IndexError:
                return
        #print("Row: ", x ,"Column: ", y , "Result: ",onerow_matrix[0][y])
        print(("Row: %7d"% (x)),("  Column: %7d"% (y)),("  Result: %1.8f"% (onerow_matrix[0][y])))
    pr = markov_matrix(onerow_matrix[0][y],n)
    result_dict[x] = pagerank_matrix(args.probability,onerow_matrix[0][y],pr)
    #print("\nResult: ",result_dict[x])

def main():

    start_time = time.time()
    file_line_number = file_len()
    
    if args.matrix:
        print("Deadling with nodes in numpy matrix mode")
        matrix_size = file_handle_matrix(file_line_number)
        try:
            original_matrix = np.zeros((matrix_size,matrix_size), dtype = float)
            fill_matrix(original_matrix,matrix_size)
            pr = markov_matrix(original_matrix,matrix_size)
            pagerank_top10(pagerank_matrix(args.probability, original_matrix, pr),matrix_size)
        except MemoryError:
            print("------------------------------------------------------------------------------------")
            print("Looks like you don't have enough RAM on your computer\n")
            print("We'll calculate it row by row\n")
            print(args.processes," processes will operate simultaneously\n")        
            print("------------------------------------------------------------------------------------")
            #Let's do it row by row
            
            result_dict = {}
            for s in range (0, int(matrix_size/args.processes) + 1):
                process_args =[]
                for k in range (0, args.processes):
                    process_args.append(k + s)
                    #print("ARGS:", process_args)
                update_progress(s / int(matrix_size/args.processes) + 1)
                pool = multiprocessing.Pool(processes=args.processes)
                func = partial(fill_matrix_by_column, matrix_size, result_dict)
                pool.map(func, process_args)
                pool.close()
                pool.join()
                
                s += args.processes
            pagerank_top10(result_dict,matrix_size)
        
        
    else:
        G = creat_network_graph()
        file_handle_graph(G,file_line_number)
        print ("Ranking...")
        pr = pagerank(G, alpha=args.probability,max_iter=2000)
        #print (pr)
        #print(type(pr))
        print("\nResult:")
        print(sorted(pr.items(), key=operator.itemgetter(1), reverse = True)[:10])
    
    print("--- Total run time: %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()

'''
References:
https://www.sharpsightlabs.com/blog/numpy-zeros-python/
https://stackoverflow.com/questions/845058/how-to-get-line-count-cheaply-in-python
https://stackoverflow.com/questions/3160699/python-progress-bar
http://ilpubs.stanford.edu:8090/422/1/1999-66.pdf
https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html
https://stackoverflow.com/questions/1053928/very-large-matrices-using-python-and-numpy
https://www.pytables.org/_modules/tables/filters.html
https://networkx.github.io/documentation/networkx-1.10/reference/classes.digraph.html
https://plot.ly/python/network-graphs/
https://blog.csdn.net/bigbennyguo/article/details/88345542
https://www.howtoforge.com/storing-files-directories-in-memory-with-tmpfs
https://www.pytables.org/usersguide/libref/file_class.html
https://stackoverflow.com/questions/25553919/passing-multiple-parameters-to-pool-map-function-in-python/25553970
https://www.python-course.eu/python3_formatted_output.php
https://airbrake.io/blog/python-exception-handling/python-indexerror
https://developers.google.com/edu/python/lists
https://stackoverflow.com/questions/25553919/passing-multiple-parameters-to-pool-map-function-in-python/25553970
https://segmentfault.com/a/1190000000711128
'''
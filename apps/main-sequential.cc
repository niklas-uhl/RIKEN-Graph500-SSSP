/*
 * main-sequential.cc
 *
 *  Created on: Mar 8, 2022
 *      Author: Daniel Rehfeldt
 */

#include <iostream>
#include <cstdlib>
#include "graphheap.hpp"
#include "shortestpath.hpp"
#include "csrgraph.hpp"

using namespace seq;

#define SEQ_PRINT_DISTANCES

int main(int argc, char* argv[] )
{
    // check number of parameters
    if( argc < 3 )
    {
       // tell the user how to run the program
       std::cerr << "Usage: " << argv[0] << " filename startnode" << '\n';
       exit(EXIT_FAILURE);
    }

    int nnodes;
    std::vector<GraphEdge> edgelist = Graph::getEdgeListFromFile(argv[1], nnodes);
    Graph graph(edgelist, nnodes);
    Sssp sssp(nnodes);

#if 0
    std::vector<int> roots;
    for( int i = 2; i < argc; i++ ) {
       roots.push_back(std::atoi(argv[i]));
       std::cout << "root: " << std::atoi(argv[i]) << '\n';
    }

    sssp.presolveSssp(graph, roots);
    return EXIT_SUCCESS;
#endif

    sssp.computeSssp(graph, std::atoi(argv[2]) - 1);

#ifdef SEQ_PRINT_DISTANCES
    const EdgeWeight* const distances = sssp.getNodeDistances();
    for( int i = 0; i < nnodes; i++ )
       std::cout << "dist[" << i + 1 << "]=" << distances[i] << '\n';
#endif

    return EXIT_SUCCESS;
}



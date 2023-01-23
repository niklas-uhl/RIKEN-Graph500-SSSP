/*
 * csrgraph.hpp
 *
 *  Created on: 07.03.2022
 *      Author: Daniel Rehfeldt
 */

#ifndef SRC_SSSP_SEQUENTIAL_CSRGRAPH_HPP_
#define SRC_SSSP_SEQUENTIAL_CSRGRAPH_HPP_

#include <vector>


namespace seq
{
   using EdgeWeight = float;

struct GraphEdge{
   EdgeWeight weight;
   int tail;
   int head;
};




class Sssp;
class Graph{
   friend Sssp;

public:
   explicit Graph(int nnodes, int nedges);
   explicit Graph(const std::vector<GraphEdge>& edgelist, int nnodes);
   ~Graph();

   static std::vector<GraphEdge> getEdgeListFromFile(const char* filename, int& nnodes);
private:

   void insertEntry(int tail, int head, EdgeWeight cost);

   int* startptr;
   int* edges_head;
   EdgeWeight* edges_cost;
   const int nnodes;
   const int nedges;
};

}


#endif /* SRC_SSSP_SEQUENTIAL_CSRGRAPH_HPP_ */

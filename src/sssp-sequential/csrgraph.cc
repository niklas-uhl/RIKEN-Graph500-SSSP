/*
 * csrgraph.cc
 *
 *  Created on: 07.03.2022
 *      Author: Daniel Rehfeldt
 */


#include "csrgraph.hpp"
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

namespace seq
{
   Graph::Graph(int nnodes, int nedges) :
      nnodes(nnodes),
      nedges(nedges)
   {
      assert(nnodes > 0 && nedges > 0);
      startptr = new int[nnodes + 1];
      edges_head = new int[2 * nedges];
      edges_cost = new EdgeWeight[2 * nedges];
   };

   Graph::Graph(const std::vector<GraphEdge>& edgelist, int nnodes) :
      Graph(nnodes, static_cast<int>(edgelist.size()))
   {
      int* node_degree = new int[nnodes]();
      assert(nedges == int(edgelist.size()));

      for( int e = 0; e < nedges; e++ )
      {
         const int tail = edgelist[e].tail;
         const int head = edgelist[e].head;
         assert(tail < nnodes && head < nnodes);

         node_degree[tail]++;
         node_degree[head]++;
      }

      startptr[0] = node_degree[0];

      for( int k = 1; k < nnodes; k++)
      {
         startptr[k] = startptr[k - 1] + node_degree[k];
      }

      startptr[nnodes] = startptr[nnodes - 1];

      // todo do we want to keep the edges in order? Here we turn them upside down...
      for( int e = 0; e < nedges; e++ )
      {
         const int tail = edgelist[e].tail;
         const int head = edgelist[e].head;
         const EdgeWeight cost = edgelist[e].weight;
         assert(tail < nnodes && head < nnodes);

         insertEntry(tail, head, cost);
         insertEntry(head, tail, cost);
      }

#ifndef NDEBUG
      assert(startptr[0] == 0);
      for( int k = 0; k < nnodes; k++)
      {
         assert(startptr[k + 1] == startptr[k] + node_degree[k]);
      }
#endif

      delete[] node_degree;
   }

   std::vector<GraphEdge> Graph::getEdgeListFromFile(const char* filename, int& nnodes)
   {
      nnodes = -1;
      int nedges;
      std::fstream file;
      file.open(filename);

      if( !file.is_open() )
      {
         std::cerr << "error while reading file \n";
         exit(EXIT_FAILURE);
      }

      std::string line;
      std::vector<GraphEdge> edgelist;

      while( std::getline(file, line) ) {

         std::istringstream iss(line);

         if( nnodes == -1 ) {
            if( !(iss >> nnodes >> nedges) ) {
               std::cerr << "error while reading file \n";
               exit(1);
            }
            assert(nedges > 0 && nnodes > 0);
            edgelist.reserve(nedges);
         }
         else
         {
            int a, b;
            EdgeWeight weight;
            if( !(iss >> a >> b >> weight) ) {
               std::cerr << "error while reading file \n";
               exit(EXIT_FAILURE);
            }

            assert(a > 0 && b > 0);
            edgelist.emplace_back(GraphEdge({weight, a - 1, b - 1}));
         }
      }

      file.close();
      return edgelist;
   }


   Graph::~Graph()
   {
      delete[] startptr;
      delete[] edges_head;
      delete[] edges_cost;
   }

   void Graph::insertEntry(int tail, int head, EdgeWeight cost)
   {
      const int pos = --startptr[tail];

      edges_head[pos] = head;
      edges_cost[pos] = cost;
   }
}

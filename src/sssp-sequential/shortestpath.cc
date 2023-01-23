/*
 * shortestpath.cc
 *
 *  Created on: Mar 9, 2022
 *      Author: Daniel Rehfeldt
 */

#include <limits>
#include "shortestpath.hpp"
#include "graphheap.hpp"
#include <iostream>


namespace seq
{
   Sssp::Sssp(int maxnnodes) :
      maxnnodes(maxnnodes),
      gheap(Graphheap<EdgeWeight>(maxnnodes))
   {
      assert(maxnnodes > 0);
      nodes_distance = new EdgeWeight[maxnnodes];
   }


   void Sssp::presolveSssp(const Graph& graph, std::vector<int> roots)
   {
      const int nnodes = graph.nnodes;
      const int nroots = roots.size();
      assert(nnodes <= maxnnodes);
      assert(nroots >= 1);

      EdgeWeight* nodes_distance_initial = new EdgeWeight[nnodes];
      int* nodes_root = new int[nnodes];
      int* nodes_pred = new int[nnodes];
      int* nodes_root_initial = new int[nnodes];
      const EdgeWeight *const cost_csr = graph.edges_cost;
      const int *const head_csr = graph.edges_head;
      const int *const start_csr = graph.startptr;

      for( int i = 0; i < nnodes; i++ ) {
         nodes_distance_initial[i] = std::numeric_limits<EdgeWeight>::max();
         nodes_distance[i] = std::numeric_limits<EdgeWeight>::max();
         nodes_root[i] = -1;
         nodes_root_initial[i] = -1;
         nodes_pred[i] = -1;
      }

      gheap.clean();

      for( int i = 0; i < nroots; i++ ) {
         const int root = roots[i];
         assert(0 <= root && root < nnodes);

         nodes_distance_initial[root] = EdgeWeight(0);
         nodes_distance[root] = EdgeWeight(0);
         nodes_root[root] = root;
         nodes_root_initial[root] = root;
         nodes_pred[root] = root;

         gheap.decreaseKey(root, EdgeWeight(0));
      }

      // note not very efficient, but should mirror parallel behavior
      for( int i = 0; i < nnodes; i++ ) {
         if( nodes_root_initial[i] != -1 )
            continue;

         for( int e = start_csr[i]; e != start_csr[i + 1]; e++ ) {
            const int head = head_csr[e];
            // todo here it should be max later on
           // if( nodes_root_initial[head] == head && (cost_csr[e] < nodes_distance_initial[i]) ) {
            if( nodes_root_initial[head] == head && (cost_csr[e] > nodes_distance_initial[i] || nodes_root_initial[i] == -1) ) {
               nodes_distance_initial[i] = cost_csr[e];
               nodes_root_initial[i] = head;
            }
         }
      }

      while( gheap.getSize() > 0 ) {
         const int k = gheap.deleteMinReturnNode();
         const int k_end = start_csr[k + 1];
         const EdgeWeight k_dist = nodes_distance[k];

         for( int e = start_csr[k]; e != k_end; e++ ) {
            const int m = head_csr[e];
            const EdgeWeight distnew = k_dist + cost_csr[e];

            if( distnew < nodes_distance_initial[m] && nodes_root_initial[m] == nodes_root[k] && k != nodes_root[k]  ) {
               nodes_distance_initial[m] = -1.0;
               nodes_pred[m] = k;
            }

            if( distnew < nodes_distance[m] ) {
               nodes_root[m] = nodes_root[k];
               nodes_distance[m] = distnew;
               gheap.decreaseKey(m, distnew);
            }
         }
      }

      int nkills = 0;
      for( int i = 0; i < nnodes; i++ ) {
         if( nodes_distance_initial[i] < -0.5 ) {
            nkills++;
            std::cout << "kill edge " << i << " -> " << nodes_root_initial[i]  <<  '\n';
         }
      }

      std::cout << "nkills=" << nkills << '\n';

      //std::cout << "nodes_pred[45]=" << nodes_pred[45]  << '\n';

      delete[] nodes_pred;
      delete[] nodes_distance_initial;
      delete[] nodes_root_initial;
      delete[] nodes_root;
   }

   void Sssp::computeSssp(const Graph& graph, int startnode)
   {
      const int nnodes = graph.nnodes;
      assert(nnodes <= maxnnodes);
      assert(0 <= startnode && startnode < nnodes);

      for( int i = 0; i < nnodes; i++ )
         nodes_distance[i] = std::numeric_limits<EdgeWeight>::max();

      gheap.clean();
      nodes_distance[startnode] = EdgeWeight(0);
      gheap.decreaseKey(startnode, EdgeWeight(0));

      const EdgeWeight *const cost_csr = graph.edges_cost;
      const int *const head_csr = graph.edges_head;
      const int *const start_csr = graph.startptr;

      while( gheap.getSize() > 0 ) {
         const int k = gheap.deleteMinReturnNode();
         const int k_end = start_csr[k + 1];
         const EdgeWeight k_dist = nodes_distance[k];

         for( int e = start_csr[k]; e != k_end; e++ ) {
            const int m = head_csr[e];
            const EdgeWeight distnew = k_dist + cost_csr[e];

            if( distnew < nodes_distance[m] ) {
              // nodes_pred[m] = k;
               nodes_distance[m] = distnew;
               gheap.decreaseKey(m, distnew);
            }
         }
      }
   }

   const EdgeWeight* Sssp::getNodeDistances() const
   {
      assert(nodes_distance);
      return nodes_distance;
   }

   Sssp::~Sssp()
   {
      delete[] nodes_distance;
   }
}

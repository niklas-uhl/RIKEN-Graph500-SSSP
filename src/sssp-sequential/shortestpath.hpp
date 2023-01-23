/*
 * shortestpath.hpp
 *
 *  Created on: Mar 9, 2022
 *      Author: Daniel Rehfeldt
 */

#ifndef SRC_SSSP_SEQUENTIAL_SHORTESTPATH_HPP_
#define SRC_SSSP_SEQUENTIAL_SHORTESTPATH_HPP_

#include "csrgraph.hpp"
#include "graphheap.hpp"


namespace seq
{

class Sssp
{

public:
   explicit Sssp(int maxnnodes);
   Sssp() = delete;

   void presolveSssp(const Graph& graph, std::vector<int> roots);
   void computeSssp(const Graph& graph, int startnode);
   const EdgeWeight* getNodeDistances() const;

   ~Sssp();

private:
   const int maxnnodes;
   EdgeWeight* nodes_distance;
   Graphheap<EdgeWeight> gheap;
};

} // namespace seq


#endif /* SRC_SSSP_SEQUENTIAL_SHORTESTPATH_HPP_ */

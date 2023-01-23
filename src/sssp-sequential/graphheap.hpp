/*
 * graphheap.hpp
 *
 *  Created on: 07.03.2022
 *      Author: Daniel Rehfeldt
 */

#ifndef SRC_SSSP_SEQUENTIAL_GRAPHHEAP_HPP_
#define SRC_SSSP_SEQUENTIAL_GRAPHHEAP_HPP_

#include <cassert>
#include <limits>

namespace seq
{

template <typename Weight>
class Graphheap
{
   static constexpr int val_unset = -1;
   static constexpr int val_set = 0;
   static constexpr Weight weight_max = std::numeric_limits<Weight>::max();
   static constexpr Weight weight_min = std::numeric_limits<Weight>::lowest();

   // heap entry
   struct HeapEntry
   {
      Weight                key;
      int                   node;
   };


public:

   explicit Graphheap(int capacity)
   {
      assert(capacity > 0);

      size = 0;
      this->capacity = capacity;
      position = new int[capacity];
      entries = new HeapEntry[capacity + 2];

      // sentinel
      entries[0].key = weight_min;

      // sentinel
      entries[capacity + 1].key = weight_max;

      this->clean();
   }

   ~Graphheap()
   {
      delete[] position;
      delete[] entries;
   }

   // cleans the heap
   void clean()
   {
      const int heap_capacity = capacity;

      size = 0;

      for( int i = 0; i < heap_capacity; i++ )
         position[i] = val_unset;
   }

   // deletes heap minimum and returns corresponding node
   int deleteMinReturnNode()
   {
      Weight fill;
      int parent;
      int hole = 1;
      int child = 2;
      int node = entries[1].node;
      const int lastentry = size--;

      assert(position[node] == 1);

      position[node] = val_set;

      // move down along min-path
      while( child < lastentry )
      {
         const Weight key1 = entries[child].key;
         const Weight key2 = entries[child + 1].key;
         assert(hole >= 1);
         assert(key1 < weight_max && key2 < weight_max);

         if( key1 > key2 )
         {
            entries[hole].key = key2;
            child++;
         }
         else
         {
            entries[hole].key = key1;
         }

         assert(entries[hole].node >= 0 && entries[hole].node < capacity);

         entries[hole].node = entries[child].node;
         position[entries[hole].node] = hole;

         hole = child;
         child *= 2;
      }

      // now hole is at last tree level, fill it with last heap entry and move it up

      fill = entries[lastentry].key;
      parent = hole / 2;

      assert(fill < weight_max && entries[parent].key < weight_max);

      while( entries[parent].key > fill )
      {
         assert(hole >= 1);

         entries[hole] = entries[parent];

         assert(entries[hole].node >= 0 && entries[hole].node < capacity);

         position[entries[hole].node] = hole;
         hole = parent;
         parent /= 2;

         assert(entries[parent].key < weight_max);
      }

      // finally, fill the hole
      entries[hole].key = fill;
      entries[hole].node = entries[lastentry].node;

      assert(entries[hole].node >= 0 && entries[hole].node < capacity);

      if( hole != lastentry )
         position[entries[hole].node] = hole;

#ifndef NDEBUG
      entries[lastentry].key = weight_max;    // set debug sentinel
#endif

      return node;
   }

   // corrects node position in heap according to new, smaller, key (or newly inserts the node) */
   void decreaseKey(int node, Weight newkey)
   {
      int hole;
      int parent;
      Weight parentkey;

      assert(newkey < weight_max && newkey > weight_min);
      assert(size <= capacity);
      assert(node >= 0 && node <= capacity);
      assert(position[node] != val_set);

      // node not yet in heap?
      if( position[node] == val_unset )
      {
         assert(size < capacity);
         hole = ++(size);
      }
      else
      {
         assert(position[node] >= 1);
         hole = position[node];

         assert(entries[hole].node == node);
         assert(entries[hole].key > newkey);
      }

      parent = hole / 2;
      parentkey = entries[parent].key;

      assert(parentkey < weight_max);

      // move hole up
      while( parentkey > newkey )
      {
         assert(hole >= 1);

         entries[hole].key = parentkey;
         entries[hole].node = entries[parent].node;
         position[entries[hole].node] = hole;
         hole = parent;
         parent /= 2;
         parentkey = entries[parent].key;
         assert(parentkey < weight_max);
      }

      // fill the hole
      entries[hole].key = newkey;
      entries[hole].node = node;
      position[node] = hole;
   }

   int getSize() { return size; }

private:
   int size;
   int capacity;
   int* position;
   HeapEntry* entries;
};


}

#endif /* SRC_SSSP_SEQUENTIAL_GRAPHHEAP_HPP_ */

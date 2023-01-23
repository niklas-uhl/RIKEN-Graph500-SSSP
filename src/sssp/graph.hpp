/*
 * graph.hpp
 *
 *  Created on: Mar 11, 2022
 *      Author: Daniel Rehfeldt
 */

#ifndef SRC_SSSP_GRAPH_HPP_
#define SRC_SSSP_GRAPH_HPP_

#include "parameters.h"


//-------------------------------------------------------------//
// 2D partitioning
//-------------------------------------------------------------//

// returns vertex owner row
int inline vertex_owner_r(int64_t v) { return v % mpi.size_2dr; }

// returns vertex owner column
int inline vertex_owner_c(int64_t v) { return (v / mpi.size_2dr) % mpi.size_2dc; }

// returns edge owner process
int inline edge_owner(int64_t v0, int64_t v1) { return vertex_owner_r(v0) + vertex_owner_c(v1) * mpi.size_2dr; }

// returns vertex owner process
int inline vertex_owner(int64_t v) { return v % mpi.size_2d; }

// returns local (on this process) representation of global vertex
int64_t inline vertex_local(int64_t v) { return v / mpi.size_2d; }

class Graph2DCSR
{
   enum {
      LOG_NBPE = PRM::LOG_NBPE,
      NBPE_MASK = PRM::NBPE_MASK
   };

public:

   Graph2DCSR() = default;

   ~Graph2DCSR()
   {
      clean();
   }

   void clean()
   {
      free(row_bitmap_); row_bitmap_ = nullptr;
      free(row_sums_); row_sums_ = nullptr;
      free(reorder_map_); reorder_map_ = nullptr;
      free(invert_map_); invert_map_ = nullptr;
      MPI_Free_mem(orig_vertexes_); orig_vertexes_ = nullptr;
      if( has_edge_bitmap_ ) {
         free(has_edge_bitmap_); has_edge_bitmap_ = nullptr;
      }
      free(is_grad1_bitmap_); is_grad1_bitmap_ = nullptr;
      free(edge_array_); edge_array_ = nullptr;
      free(edge_weight_array_); edge_weight_array_ = nullptr;
      if( edge_head_ownerc_ ) {
         free(edge_head_ownerc_); edge_head_ownerc_= nullptr;
      }
      free(row_starts_); row_starts_ = nullptr;
      free(row_starts_heavy_); row_starts_heavy_ = nullptr;
      free(vertices_minweight_); vertices_minweight_= nullptr;
   }

   int pred_size() const { return num_orig_local_verts_; }

   int log_orig_global_verts() const { return log_orig_global_verts_; }

   // Reference Functions
   static int rank(int r, int c) { return c * mpi.size_2dr + r; }
   int64_t swizzle_vertex(int64_t v) {
      return SeparatedId(vertex_owner(v), vertex_local(v), local_bits_).value;
   }
   int64_t unswizzle_vertex(int64_t v) const {
      SeparatedId id(v);
      return id.high(local_bits_) + id.low(local_bits_) * num_local_verts_;
   }

   // vertex id converter
   SeparatedId VtoD(int64_t v) const {
      return SeparatedId(vertex_owner_r(v), vertex_local(v), local_bits_);
   }
   SeparatedId VtoS(int64_t v) const {
      return SeparatedId(vertex_owner_c(v), vertex_local(v), local_bits_);
   }
   int64_t DtoV(SeparatedId id, int c) const {
      return id.low(local_bits_) * mpi.size_2d + rank(id.high(local_bits_), c);
   }
   int64_t StoV(SeparatedId id, int r) const {
      return id.low(local_bits_) * mpi.size_2d + rank(r, id.high(local_bits_));
   }
   SeparatedId StoD(SeparatedId id, int r) const {
      return SeparatedId(r, id.low(local_bits_), local_bits_);
   }
   SeparatedId DtoS(SeparatedId id, int c) const {
      return SeparatedId(c, id.low(local_bits_), local_bits_);
   }
   int get_weight_from_edge(int64_t e) const {
      return e & ((1 << log_max_weight_) - 1);
   }

   bool has_edge(int64_t v, bool has_weight = false) const {
      if(vertex_owner(v) == mpi.rank_2d) {
         assert(has_edge_bitmap_);
         int64_t v_local = reorder_map_[v / mpi.size_2d];
         if(v_local > num_local_verts_) return false;
         int64_t word_idx = v_local >> LOG_NBPE;
         int bit_idx = v_local & NBPE_MASK;
         return has_edge_bitmap_[word_idx] & (BitmapType(1) << bit_idx);
      }
      return false;
   }

   bool local_vertex_isDeg1(uint64_t v) const {
      const uint64_t base = v >> LOG_NBPE;
      const uint64_t shift = v & NBPE_MASK;
      assert(base == v / PRM::NBPE);
      assert(shift == v % PRM::NBPE);
      return (is_grad1_bitmap_[base] & uint64_t(1) << shift);
   }

   // separates heavy edges from light ones
   void separateHeavyEdges(float delta_step) {
      const int64_t num_local_verts = num_local_verts_;
      const int64_t local_bitmap_width = num_local_verts / (PRM::NBPE);
      const int64_t row_bitmap_length = local_bitmap_width * mpi.size_2dc;
      const int64_t non_zero_rows = row_sums_[row_bitmap_length];
      assert(!row_starts_heavy_);
      assert(0 < delta_step && delta_step <= 1.0);
      row_starts_heavy_ =  (int64_t*)cache_aligned_xmalloc(non_zero_rows*sizeof(row_starts_heavy_[0]));

      if( mpi.isMaster() ) print_with_prefix("Separating heavy edges.");
      for( int64_t non_zero_idx = 0; non_zero_idx < non_zero_rows; ++non_zero_idx ) {
         const int64_t e_start = row_starts_[non_zero_idx];
         const int64_t e_end = row_starts_[non_zero_idx + 1];
         const int64_t e_length = e_end - e_start;
         assert(e_length > 0);

         int64_t* edges = (int64_t*)cache_aligned_xmalloc(e_length * sizeof(*edges));
         float* weights = (float*)cache_aligned_xmalloc(e_length * sizeof(*weights));
         uint16_t* owners = (uint16_t*)cache_aligned_xmalloc(e_length * sizeof(*owners));
         if( edges == nullptr || weights == nullptr || owners == nullptr ) {
            printf("Out of memory while trying to allocate temporary edge array");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
         }

         assert(sizeof(*edges) == sizeof(*edge_array_) && sizeof(*weights) == sizeof(*edge_weight_array_) && sizeof(*owners) == sizeof(*edge_head_ownerc_));
         memcpy(edges, edge_array_+ e_start, e_length * sizeof(*edges));
         memcpy(weights, edge_weight_array_ + e_start, e_length * sizeof(*weights));
         memcpy(owners, edge_head_ownerc_ + e_start, e_length * sizeof(*owners));
         int64_t nheavy = 0;
         for( int64_t i = 0; i < e_length; i++ ) {
            assert(weights[i] >= 0.0);
            if( weights[i] > delta_step ) {
               nheavy++;
               weights[i] *= -1.0;
            }
         }
         row_starts_heavy_[non_zero_idx] = e_end - nheavy;

         int64_t start_heavy = row_starts_heavy_[non_zero_idx];
         int64_t start_light = e_start;
         for( int64_t i = 0; i < e_length; i++ ) {
            if( weights[i] < 0.0 ) {
               edge_weight_array_[start_heavy] = -weights[i];
               edge_head_ownerc_[start_heavy] = owners[i];
               edge_array_[start_heavy++] = edges[i];
            }
            else {
               edge_weight_array_[start_light] = weights[i];
               edge_head_ownerc_[start_light] = owners[i];
               edge_array_[start_light++] = edges[i];
            }
         }
         assert(start_light == row_starts_heavy_[non_zero_idx]);
         assert(start_heavy == e_end);

#ifndef NDEBUG
        for( int64_t i = e_start; i < row_starts_heavy_[non_zero_idx]; i++ )
           assert(comp::isLE(edge_weight_array_[i], delta_step));
        for( int64_t i = row_starts_heavy_[non_zero_idx]; i < e_end; i++ )
           assert(edge_weight_array_[i] > delta_step);
#endif

         free(owners);
         free(weights);
         free(edges);
      }
      MPI_Barrier(mpi.comm_2d);

      if( mpi.isMaster() ) print_with_prefix("Finished separating heavy edges.");
   }


   // writes locally saved graph to file; just for testing!
   void writeLocalToFile(const char* filepath) const
   {
      std::ofstream outfile;
      outfile.open(filepath);
      const int64_t num_global_verts = num_local_verts_ * mpi.size_2dc;

#if 0
      std::cout << "reorder_map_: " << std::endl;
      for( int i = 0; i < num_global_verts_; i++ )
         std::cout << reorder_map_[i]  << std::endl;

      std::cout << "invert_map_: " << std::endl;
      for( int i = 0; i < num_global_verts_; i++ )
         std::cout << invert_map_[i]  << std::endl;
#endif

      const int lgl = local_bits_;
      const int r_mask = (1 << r_bits_) - 1;
      outfile << "num_local_verts_=" << num_local_verts_ << " lgl=" << lgl << " r_mask=" << r_mask << '\n';

      for( int64_t i = 0; i < num_global_verts; i++ )
      {
         const uint64_t pos = i / 64;
         const BitmapType row_bitmap_i = row_bitmap_[pos];
         const uint64_t rest = i % 64;

         if( row_bitmap_i & (BitmapType(1) << rest) )
         {
            const int start = row_sums_[pos] + __builtin_popcountl(row_bitmap_i & ((BitmapType(1) << rest) - 1));

            outfile << "vertex=" << i;
            outfile << " degree=" << row_starts_[start + 1] - row_starts_[start]  << '\n';

            for(int64_t e = row_starts_[start]; e < row_starts_[start + 1]; ++e) {
               //const int dest = (edge_array_[e] >> lgl) & r_mask;
               const int head = edge_array_[e] & ((uint32_t(1) << lgl) - 1);
               outfile << "...head=" << head << " weight=" << edge_weight_array_[e] << '\n';
            }
         }
      }

      outfile.close();
   }

   // Array Indices:
   //  - Compressed Source Index (CSI) : source index skipping vertices with no edges in this rank
   //  - Source Bitmap Index (SBI) : source index / 64
   //   - Pred: same as Pred (original local vertices)

   // num_local_verts_ <= num_orig_local_verts_ <= length(reorder_map_)

   BitmapType* row_bitmap_ = nullptr; // Index: SBI
   TwodVertex* row_sums_ = nullptr; // Index: SBI
   BitmapType* has_edge_bitmap_ = nullptr; // for every local vertices, Index: SBI
   BitmapType* is_grad1_bitmap_ = nullptr; // for every local vertices, Index: SBI
   LocalVertex* reorder_map_ = nullptr; // Index: Pred
   LocalVertex* invert_map_ = nullptr; // Index: Reordered Pred
   LocalVertex* orig_vertexes_ = nullptr; // Index: CSI
   float* vertices_minweight_ = nullptr; // minimum incident edge weight

   int64_t* edge_array_ = nullptr;
   float* edge_weight_array_ = nullptr;
   uint16_t* edge_head_ownerc_ = nullptr;
   int64_t* row_starts_ = nullptr; // Index: CSI
   int64_t* row_starts_heavy_ = nullptr; // where the heavy edges start

   int log_orig_global_verts_ = 0; // estimated SCALE parameter
   int log_max_weight_ = 0;
   int64_t num_orig_local_verts_ = 0; // number of local vertices for original graph: same number per process, but rounded up

   int max_weight_ = 0;
   int64_t num_global_edges_ = 0; // number of edges after reduction of hyper edges
   int64_t num_global_verts_ = 0; // number of vertices that have at least one edge

   int local_bits_ = 0; // local bits for computation
   int orig_local_bits_ = 0; // local bits for original vertex id
   int r_bits_ = 0;
   int64_t num_local_verts_ = 0; // number of local vertices for computation: maximum among all non-zero vertices on all processes
};


#endif /* SRC_SSSP_GRAPH_HPP_ */

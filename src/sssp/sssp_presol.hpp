/*
 * sssp_presol.hpp
 *
 *  Created on: Jun 22, 2022
 *      Author: Daniel Rehfeldt
 */

#ifndef SRC_SSSP_SSSP_PRESOL_HPP_
#define SRC_SSSP_SSSP_PRESOL_HPP_

#include "sssp.hpp"
#include <string>

class SsspPresolver {
   using Graph = Graph2DCSR;

public:
   constexpr static float deletion_weight = 2.0;
   SsspPresolver(SsspBase& sssp) :
      sssp_(sssp),
      graph_(sssp.graph_),
      root_local_(0),
      has_root_local_(false),
      root_round_(-1),
      current_round_(-1),
      max_n_rounds_(0),
      nkilled_all_(-1),
      n_all_(-1)
   {}

   ~SsspPresolver()
   {}

   void presolve_sssp(int niterations, int64_t* pred, float* dist);

private:

   // runs new round
   void presolve_round_run()
   {
   #if VERBOSE_MODE
      using namespace profiling;

      const double start_time = MPI_Wtime();
      expand_settled_bitmap_time = expand_buckets_time = expand_time = fold_time = 0.0;
      total_edge_top_down = total_edge_bottom_up = 0;
      g_tp_comm = g_bu_pred_comm = g_bu_bitmap_comm = g_bu_list_comm = g_expand_bitmap_comm = g_expand_list_comm = 0;
   #endif

      sssp_.initialize_sssp_run();
      assert(sssp_.prev_buckets_sizes.size() == 0);

   #if VERBOSE_MODE
      if(mpi.isMaster()) print_with_prefix("Time of initialize: %f ms", (MPI_Wtime() - start_time) * 1000.0);
   #endif

      sssp_.cq_root_list_ = (int64_t*) page_aligned_xmalloc(sssp_.cq_distance_buf_length_ * sizeof(*sssp_.cq_root_list_));
      sssp_.nq_root_list_ = (int64_t*) page_aligned_xmalloc(sssp_.nq_buf_length_ * sizeof(*sssp_.nq_root_list_));
      sssp_.dist_presol_ = (float*) cache_aligned_xmalloc(graph_.pred_size()*sizeof(*sssp_.dist_presol_));
      sssp_.pred_presol_ = (int64_t*) cache_aligned_xmalloc(graph_.pred_size()*sizeof(*sssp_.pred_presol_));
      for( int i = 0; i < graph_.pred_size(); i++ ) {
         sssp_.dist_presol_[i] = std::numeric_limits<float>::max();
         sssp_.pred_presol_[i] = -1;
      }
      const double delta_step_org = sssp_.delta_step_;
      sssp_.delta_step_ = 0.0;
      sssp_.current_level_ = -2;
      sssp_.is_light_phase_ = false;
      sssp_.is_presolve_mode_ = true;

      expand_roots(true);

      // set the initial dist_presol distances, namely edge-weights to adjacent vertices of the roots
      sssp_.run_sssp_phases();
      sssp_.clear_nq_stack();
      sssp_.delta_step_ = delta_step_org;

      sssp_.initialize_sssp_run();
      expand_roots(false);

      sssp_.execute_sssp_run(std::numeric_limits<int64_t>::max());

      int todo;
      const uint64_t num_local_verts = uint64_t(graph_.num_local_verts_);
      int64_t n_killed = 0;
      int64_t n_all = 0;

      for( uint64_t i = 0; i < num_local_verts; i++ ) {
         if( sssp_.dist_presol_[i] < comp::infinity ) {
            n_all++;

           // if( sssp_.dist_[i] == 0.0 )  std::cout << mpi.rank_2d << " 0.0: " << i  << " " << sssp_.dist_[i] << " <? "<< sssp_.dist_presol_[i] << '\n';

            //if( comp::isLT(sssp_.dist_[i], sssp_.dist_presol_[i]) ) {
            if( comp::isEQ(sssp_.dist_presol_[i], -1.0) ) {
               n_killed++;
               //std::cout << "kill" << graph_.invert_map_[i] * mpi.size_2d + mpi.rank_2d << " root=" << sssp_.pred_presol_[i]  << '\n';
            }
         }
      }

     // std::cout << mpi.rank_2d << " n_all=" << n_all << " n_killed="  << n_killed  << '\n';


      int64_t send = n_killed;
      MPI_Reduce(&send, &n_killed, 1, MpiTypeOf<int64_t>::type, MPI_SUM, 0, mpi.comm_2d);
      send = n_all;
      MPI_Reduce(&send, &n_all, 1, MpiTypeOf<int64_t>::type, MPI_SUM, 0, mpi.comm_2d);

      if( mpi.isMaster() && n_all > 0)
         std::cout << " n_all=" << n_all << " n_killed="  << n_killed << " ratio=" << double(n_killed) / n_all << '\n';

      nkilled_all_ += n_killed;
      n_all_ += n_all;

      mark_presolve_roots_bitmap();
      sssp_.expand_vertices_bitmap(sssp_.vertices_isSettledLocal_, sssp_.vertices_isSettled_);

      int64_t nq_size = 0;
      expand_dominated_top_down(nq_size);

      if( nq_size > 0 ) {
         top_down_mark_redundant_edges();

         BitmapType* q_row_sums = expand_dominated_bottom_up();
         bottom_up_mark_redundant_edges(q_row_sums);

         free(q_row_sums);

         delete_marked_edges();
      }

      free(sssp_.cq_root_list_); sssp_.cq_root_list_ = nullptr;
      free(sssp_.nq_root_list_); sssp_.nq_root_list_ = nullptr;
      free(sssp_.dist_presol_); sssp_.dist_presol_ = nullptr;
      free(sssp_.pred_presol_); sssp_.pred_presol_ = nullptr;
   }

   // removes marked edges
   void delete_marked_edges() {
      const int64_t num_local_verts = graph_.num_local_verts_;
      const int64_t local_bitmap_width = num_local_verts / (PRM::NBPE);
      const uint64_t row_bitmap_length = local_bitmap_width * mpi.size_2dc;
      const int64_t non_zero_rows_org = graph_.row_sums_[row_bitmap_length];
      const int64_t nedges_org = graph_.row_starts_[non_zero_rows_org];

      // adapt row starts
      int64_t shift = 0;
      for( int64_t non_zero_idx = 0; non_zero_idx < non_zero_rows_org; ++non_zero_idx ) {
         const int64_t e_start = graph_.row_starts_[non_zero_idx];
         const int64_t e_start_heavy = graph_.row_starts_heavy_[non_zero_idx];
         const int64_t e_end = graph_.row_starts_[non_zero_idx + 1];
         graph_.row_starts_[non_zero_idx] -= shift;

         for( int64_t e = e_start; e < e_start_heavy; ++e )
            if( comp::isEQ(graph_.edge_weight_array_[e], deletion_weight) )
               shift++;

         graph_.row_starts_heavy_[non_zero_idx] -= shift;

         for( int64_t e = e_start_heavy; e < e_end; ++e )
            if( comp::isEQ(graph_.edge_weight_array_[e], deletion_weight) )
               shift++;
      }
      graph_.row_starts_[non_zero_rows_org] -= shift;

      assert(num_local_verts % 64 == 0);
      assert(num_local_verts >> PRM::LOG_NBPE == num_local_verts / 64);
      assert(64 == PRM::NBPE);

      for( uint64_t word = 0; word < row_bitmap_length; word++ ) {
         const BitmapType row_bitmap_word = graph_.row_bitmap_[word];
         for( uint64_t bit = 0; bit < 64; bit++) {
            if( row_bitmap_word & (BitmapType(1) << bit) ) {
               const TwodVertex start = graph_.row_sums_[word] + __builtin_popcountl(row_bitmap_word & ((BitmapType(1) << bit) - 1));

               if( graph_.row_starts_[start] == graph_.row_starts_[start + 1] ) {
                  graph_.row_bitmap_[word] ^= (BitmapType(1) << bit);
                  assert(!(graph_.row_bitmap_[word] & (BitmapType(1) << bit)));
               }
            }
         }
      }

      int64_t non_zero_rows_new = 0;

      // clean row starts
      for( int64_t non_zero_idx = 0; non_zero_idx < non_zero_rows_org; ++non_zero_idx ) {
         const int64_t e_start = graph_.row_starts_[non_zero_idx];
         const int64_t e_end = graph_.row_starts_[non_zero_idx + 1];

         if( e_start != e_end ) {
            graph_.orig_vertexes_[non_zero_rows_new] = graph_.orig_vertexes_[non_zero_idx];
            graph_.row_starts_[non_zero_rows_new] = graph_.row_starts_[non_zero_idx];
            graph_.row_starts_heavy_[non_zero_rows_new] = graph_.row_starts_heavy_[non_zero_idx];
            non_zero_rows_new++;
         }
      }
      assert(non_zero_rows_new <= non_zero_rows_org);
      graph_.row_starts_[non_zero_rows_new] = graph_.row_starts_[non_zero_rows_org];

      // rebuild row sums and offsets
      const int64_t src_bitmap_size = (graph_.num_local_verts_ / PRM::NBPE) * mpi.size_2dc;
      graph_.row_sums_[0] = 0;
      for( int64_t i = 0; i < src_bitmap_size; ++i ) {
         const int num_rows = __builtin_popcountl(graph_.row_bitmap_[i]);
         graph_.row_sums_[i+1] = graph_.row_sums_[i] + num_rows;
      }

      // now adapt the actual edges-_
      int64_t nedges_new = 0;
      for( int64_t e = 0; e < nedges_org; ++e ) {
         if( comp::isEQ(graph_.edge_weight_array_[e], deletion_weight) )
            continue;

         graph_.edge_weight_array_[nedges_new] = graph_.edge_weight_array_[e];
         graph_.edge_array_[nedges_new] = graph_.edge_array_[e];
         graph_.edge_head_ownerc_[nedges_new] = graph_.edge_head_ownerc_[e];
         nedges_new++;
      }
      assert(shift == nedges_org - nedges_new);
      assert(nedges_new == graph_.row_starts_[non_zero_rows_new]);
    //  std::cout << mpi.rank_2d << " nkills=" <<  nedges_org - nedges_new <<  '\n';
   }


   // creates list of presolving roots
   int roots_make_nq_list(bool is_first_expansion) {
      const TwodVertex shifted_rc = TwodVertex(mpi.rank_2dc) << graph_.local_bits_;

      assert(!sssp_.next_bitmap_or_list_);
      int result_size = 0;
      int todo; // can we have the offset automatic?
      const int offset = 40;//;80;//22;
      has_root_local_ = false;

      if( (mpi.size_2d > 1 && (current_round_ + mpi.rank_2d) % offset != 0)  ) {
         return result_size;
      }

      //assert(root_round_ <= current_round_);
      const int root = (int64_t(root_round_) + 10 * int64_t(mpi.rank_2d)) % int64_t(max_n_rounds_ / offset);
      assert(root >= 0);

      if( !is_first_expansion )
         root_round_++;

      if( graph_.local_vertex_isDeg1(root) ) {
         //assert(0); // here we should just try another on maybe...
         return result_size;
      }

      const int64_t root_global = (int64_t (graph_.invert_map_[root]) * int64_t(mpi.size_2d)) + int64_t (mpi.rank_2d);

      //if( graph_.reorder_map_[vertex_local(root_global)] != (LocalVertex) root )
        // std::cout << "rank" << mpi.rank_2d << " XX FAIL root_local=" << root << " graph_.invert_map_[root]=" << graph_.invert_map_[root] <<  " root_global=" << root_global << " local=" << vertex_local(root_global) << " reordered local=" << graph_.reorder_map_[vertex_local(root_global)]  << '\n';

      assert(graph_.reorder_map_[vertex_local(root_global)] == (LocalVertex) root);

      //std::cout << "rank" << mpi.rank_2d << " pos=" << root << " root=" << root_global << '\n';

      has_root_local_ = true;
      root_local_ = root;
      sssp_.pred_[root] = root_global;
      sssp_.dist_[root] = 0.0;
      sssp_.nq_list_[0] = root | shifted_rc;
      sssp_.nq_root_list_[0] = root_global;
      sssp_.nq_distance_list_[0] = 0.0;
      result_size = 1;

      return result_size;
   }

   // expands the root nodes for the next presolving iteration
   void expand_roots(bool is_first_expansion) {
      sssp_.reset_root_grad1 = false;
      const int nq_size = roots_make_nq_list(is_first_expansion);
      sssp_.top_down_expand_nq(nq_size);
   }

   int dominated_make_nq_list_(TwodVertex shifted_rc) {
      const int max_threads = omp_get_max_threads();
      int threads_offset[max_threads + 1];
      int result_size = -1;
      //const float* const restrict dist = sssp_.dist_;

#pragma omp parallel
      {
         const uint64_t num_local_verts = uint64_t(graph_.num_local_verts_);
         const int tid = omp_get_thread_num();
         int count = 0;

#pragma omp for schedule(static) nowait
         for( uint64_t i = 0; i < num_local_verts; i++ ) {
            //if( comp::isLT(sssp_.dist_presol_[i], comp::infinity) && comp::isLT(dist[i], sssp_.dist_presol_[i]) && sssp_.pred_[i] == sssp_.pred_presol_[i] )
            if( comp::isEQ(sssp_.dist_presol_[i], -1.0) )
               count+=2;
         }
         threads_offset[tid + 1] = count;
#pragma omp barrier
#pragma omp single
         {
            threads_offset[0] = 0;
            for( int i = 0; i < max_threads; i++ )
               threads_offset[i + 1] += threads_offset[i];

            result_size = threads_offset[max_threads];
            sssp_.update_nq_capacity(result_size);
         } // barrier

#ifndef NDEBUG
#pragma omp for schedule(static)
         for(int i = 0; i < result_size; ++i)
            sssp_.nq_list_[i] = num_local_verts;
#endif
         int offset = threads_offset[tid];
#pragma omp for schedule(static) nowait
         for( uint64_t i = 0; i < num_local_verts; i++ ) {
             if( comp::isEQ(sssp_.dist_presol_[i], -1.0) ) {
            //if( comp::isLT(sssp_.dist_presol_[i], comp::infinity) && comp::isLT(dist[i], sssp_.dist_presol_[i]) &&  ) {
               assert(sssp_.nq_list_[offset] == num_local_verts);
              //std::cout <<  mpi.rank_2d  <<  " for cq=" << (i | shifted_rc)  << " sends vertex " << i << " or " <<  (i | shifted_rc) << " org=" << graph_.invert_map_[i] * mpi.size_2d + mpi.rank_2d  << " root=" << sssp_.pred_[i]  << '\n';
              //std::cout << "sending " << (i | shifted_rc) << " " << sssp_.pred_presol_[i] << '\n';

               sssp_.nq_list_[offset++] = i | shifted_rc;
               sssp_.nq_list_[offset++] = sssp_.pred_presol_[i];
            }
         }

         // todo remove
         if( offset != threads_offset[tid + 1] ) {
            std::cout << "dominated_make_nq_list_top_down issue \n";
            print_with_prefix("dominated_make_nq_list_top_down issue!");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
         }
      }

      return result_size;
   }


   // marks presolving roots
   void mark_presolve_roots_bitmap() {
      BitmapType* const restrict is_root = (BitmapType*)sssp_.vertices_isSettledLocal_;

      memory::clean_mt(sssp_.vertices_isSettled_, sssp_.get_bitmap_size_local() * mpi.size_2dr * sizeof(*sssp_.vertices_isSettled_));
      memory::clean_mt(is_root, sssp_.get_bitmap_size_local() * sizeof(*is_root));

      if( has_root_local_ ) {
         int todo; // remove this method etc
         //std::cout << "local root=" << root_local_ << '\n';
         const uint64_t word = root_local_ >> SsspBase::LOG_NBPE;
         const uint64_t bit = root_local_ & SsspBase::NBPE_MASK;
         is_root[word] |= BitmapType(1) << bit;
      }
   }



   // expands dominated vertices (and predecessors) during presolving
   BitmapType* expand_dominated_bottom_up() {
      assert(!sssp_.next_bitmap_or_list_);
      const TwodVertex shifted_r = TwodVertex(mpi.rank_2dr) << graph_.local_bits_;
      const int nq_size = dominated_make_nq_list_(shifted_r);



#if 1
      int todo; // deleteme
      int64_t global_nq_size = 0;
      int64_t send_nq_size = nq_size;
      int64_t nq_sum;
      MPI_Allreduce(&send_nq_size, &nq_sum, 1, MpiTypeOf<int64_t>::type, MPI_SUM, mpi.comm_2d);
      global_nq_size = nq_sum;

      if( mpi.isMaster() )
         std::cout << "number of BOTTOM-UP eliminations: " << global_nq_size << '\n';
#endif

      const int comm_size = mpi.comm_c.size;
      int recv_size[comm_size];
      int recv_off[comm_size + 1];
      MPI_Allgather(&nq_size, 1, MPI_INT, recv_size, 1, MPI_INT, mpi.comm_c.comm);
      recv_off[0] = 0;
      for(int i = 0; i < comm_size; ++i)
         recv_off[i + 1] = recv_off[i] + recv_size[i];
      const int settled_size = recv_off[comm_size];

      assert(settled_size % 2 == 0);
      sssp_.update_work_buf(settled_size * int64_t(sizeof(TwodVertex)));
      TwodVertex* recv_buf = (TwodVertex*) sssp_.work_buf_;

#if ENABLE_MY_ALLGATHER == 1
      MpiCol::my_allgatherv(sssp_.nq_list_, nq_size, recv_buf, recv_size, recv_off, mpi.comm_c);
#elif ENABLE_MY_ALLGATHER == 2
      my_allgatherv_2d(sssp_.nq_list_, nq_size, recv_buf, recv_size, recv_off, mpi.comm_c);
#else
      MPI_Allgatherv(sssp_.nq_list_, nq_size, MpiTypeOf<TwodVertex>::type, recv_buf, recv_size, recv_off, MpiTypeOf<TwodVertex>::type, mpi.comm_c.comm);
#endif

      const int lgl = graph_.local_bits_;
      const uint32_t local_mask = (uint32_t(1) << lgl) - 1;
      const int64_t num_local_verts = graph_.num_local_verts_;

      memory::clean_mt(sssp_.vertices_isSettled_, sssp_.get_bitmap_size_local() * mpi.size_2dr * sizeof(*(sssp_.vertices_isSettled_)));

#pragma omp parallel for schedule(static)
      for( int i = 0; i < settled_size; i += 2 ) {
         const SeparatedId src(recv_buf[i]);
         const TwodVertex src_c = src.value >> lgl;
         const TwodVertex compact = src_c * num_local_verts + (src.value & local_mask);
         const TwodVertex word_idx = compact >> PRM::LOG_NBPE;
         const int bit_idx = compact & PRM::NBPE_MASK;

         assert(word_idx < uint64_t(sssp_.get_bitmap_size_local() * mpi.size_2dr));
         assert(!(sssp_.vertices_isSettled_[word_idx] & (uint64_t(1) << bit_idx)));
#pragma omp atomic update
         sssp_.vertices_isSettled_[word_idx] |= (uint64_t(1) << bit_idx);
      }

      const int64_t q_size = sssp_.get_bitmap_size_local() * mpi.size_2dr;
      BitmapType* q_row_sums = (BitmapType*)cache_aligned_xmalloc((q_size + 1) * sizeof(*q_row_sums));

      q_row_sums[0] = 0;
      for(int64_t i = 0; i < q_size; ++i) {
         q_row_sums[i+1] = q_row_sums[i] + __builtin_popcountl(sssp_.vertices_isSettled_[i]);
      }

      return q_row_sums;
   }


   // expands dominated vertices (and predecessors) during presolving
   void expand_dominated_top_down(int64_t& global_nq_size) {
      assert(!sssp_.next_bitmap_or_list_);

      const TwodVertex shifted_c = TwodVertex(mpi.rank_2dc) << graph_.local_bits_;
      const int nq_size = dominated_make_nq_list_(shifted_c);

      int64_t send_nq_size = nq_size;
      int64_t nq_sum;
      MPI_Allreduce(&send_nq_size, &nq_sum, 1, MpiTypeOf<int64_t>::type, MPI_SUM, mpi.comm_2d);
      global_nq_size = nq_sum;

      if( global_nq_size == 0 ) {
         assert(nq_size == 0);
         return;
      }

      if( mpi.isMaster() ) {
         std::cout << "number of eliminations: " << global_nq_size << '\n';
      }

      const int comm_size = mpi.comm_r.size;
      int recv_size[comm_size];
      int recv_off[comm_size + 1];
      MPI_Allgather(&nq_size, 1, MPI_INT, recv_size, 1, MPI_INT, mpi.comm_r.comm);
      recv_off[0] = 0;
      for(int i = 0; i < comm_size; ++i)
         recv_off[i + 1] = recv_off[i] + recv_size[i];
      sssp_.cq_size_ = recv_off[comm_size];

      sssp_.update_work_buf(int64_t(sssp_.cq_size_) * int64_t(sizeof(TwodVertex)));
      TwodVertex* recv_buf = (TwodVertex*) sssp_.work_buf_;

#if ENABLE_MY_ALLGATHER == 1
      MpiCol::my_allgatherv(nq_list_, nq_size, recv_buf, recv_size, recv_off, mpi.comm_r);
#elif ENABLE_MY_ALLGATHER == 2
      my_allgatherv_2d(nq_list_, nq_size, recv_buf, recv_size, recv_off, mpi.comm_r);
#else
      MPI_Allgatherv(sssp_.nq_list_, nq_size, MpiTypeOf<TwodVertex>::type, recv_buf, recv_size, recv_off, MpiTypeOf<TwodVertex>::type, mpi.comm_r.comm);
#endif
      sssp_.cq_any_ = recv_buf;
      sssp_.work_buf_state_ = SsspBase::Work_buf_state::cq;
   }



   void bottom_up_mark_redundant_edges(const BitmapType* q_row_sums) {
      int64_t nkilled = 0;

      std::vector<int64_t> roots(mpi.size_2dc);
      std::vector<int64_t> roots_org(mpi.size_2dc);
      int64_t my_root = -1;
      int64_t my_root_org = -1;

      if( has_root_local_ ) {
         const TwodVertex shifted_c = TwodVertex(mpi.rank_2dc) << graph_.local_bits_;
         my_root = root_local_ | shifted_c;
         my_root_org = (int64_t (graph_.invert_map_[root_local_]) * int64_t(mpi.size_2d)) + int64_t (mpi.rank_2d);
      }

      MPI_Allgather(&my_root, 1, get_mpi_type(my_root), roots.data(), 1, get_mpi_type(my_root), mpi.comm_2dr);
      MPI_Allgather(&my_root_org, 1, get_mpi_type(my_root_org), roots_org.data(), 1, get_mpi_type(my_root_org), mpi.comm_2dr);


#pragma omp parallel
      {
         SET_OMP_AFFINITY;
         int64_t* const restrict edge_array = graph_.edge_array_;
         float* const restrict edge_weight_array = graph_.edge_weight_array_;
         const int lgl = graph_.local_bits_;
         const uint32_t local_mask = (uint32_t(1) << lgl) - 1;
         const int64_t L = graph_.num_local_verts_;
         const int r_bits = graph_.r_bits_;
//         const uint32_t row_mask = (uint32_t(1) << r_bits) - 1;

         {
            const TwodVertex* restrict const recv_buf = (TwodVertex*) sssp_.work_buf_;
            assert(sssp_.cq_size_ % 2 == 0);
#pragma omp for
            for( size_t i = 0; i < roots.size(); i++ ) {
               if( roots[i] == -1 )
                  continue;
               assert(roots_org[i] >= 0);
             //  std::cout << "checking root=" << roots[i] << '\n';
               const SeparatedId src(roots[i]);
               const TwodVertex src_c = src.value >> lgl;
               const TwodVertex compact = src_c * L + (src.value & local_mask);
               const TwodVertex word_idx = compact >> SsspBase::LOG_NBPE;
               const int bit_idx = compact & SsspBase::NBPE_MASK;
               const BitmapType row_bitmap_i = graph_.row_bitmap_[word_idx];
               const BitmapType mask = BitmapType(1) << bit_idx;

               if(row_bitmap_i & mask) {
                  const BitmapType low_mask = (BitmapType(1) << bit_idx) - 1;
                  const TwodVertex non_zero_off = graph_.row_sums_[word_idx] + __builtin_popcountl(graph_.row_bitmap_[word_idx] & low_mask);
                  const int64_t e_start = graph_.row_starts_[non_zero_off];
                  const int64_t e_end = graph_.row_starts_[non_zero_off + 1];

                  for( int64_t e = e_start; e < e_end; ++e ) {
                     const int64_t tgt = edge_array[e];
#if 0
                     const uint32_t head_unordered = tgt >> (lgl + graph_.r_bits_);
                     const uint32_t row = (tgt >> lgl) & row_mask;
                     const uint32_t owner = (mpi.size_2dr * graph_.edge_head_ownerc_[e] + row);
                     const TwodVertex head_global = int64_t(mpi.size_2d) * int64_t(head_unordered) + int64_t(owner);
#endif

                     if( sssp_.top_down_target_is_settled(tgt, r_bits, lgl, L) ) {
                        const BitmapType bit_idx = SeparatedId(SeparatedId(tgt).low(r_bits + lgl)).compact(lgl, L);
                        const BitmapType word_tgt = bit_idx >> PRM::LOG_NBPE;
                        assert((bit_idx & PRM::NBPE_MASK) < 64);
                        const BitmapType mask_x = (BitmapType(1) << (bit_idx & PRM::NBPE_MASK)) - 1;
                        const TwodVertex pos = 2 * (q_row_sums[word_tgt] + __builtin_popcountl(sssp_.vertices_isSettled_[word_tgt] & mask_x));
                        assert(pos + 1 < sssp_.work_buf_size_ / sizeof(TwodVertex));

                      //  std::cout << "word_tgt=" << wordtgt << '\n';
                      //  std::cout << " pos=" << pos << " " <<  recv_buf[pos + 1] << " vs " <<  head_global << '\n';

                        if( recv_buf[pos + 1] != TwodVertex(roots_org[i]) )
                           continue;

                     //   std::cout << mpi.rank_2d << " BOTTOM-UP for local: " << (src.value & local_mask)  << "->" << head_unordered << " delete: " << (src.value & local_mask) << "->" << (head_global) << " c=" << edge_weight_array[e] << '\n';
                        edge_weight_array[e] = deletion_weight;

#pragma omp atomic update
                        nkilled++;
                     }
                  }
               } // if(row_bitmap_i & mask) {
            } // #pragma omp for // implicit barrier
         }
      }

      int64_t send = nkilled;
      MPI_Reduce(&send, &nkilled, 1, MpiTypeOf<int64_t>::type, MPI_SUM, 0, mpi.comm_2d);
      if( mpi.isMaster() )
         std::cout << "BOTTOM-UP REAL nkilled=" << nkilled << '\n';
   }


   void top_down_mark_redundant_edges() {
      int64_t nkilled = 0;

#pragma omp parallel
      {
         SET_OMP_AFFINITY;
         int64_t* const restrict edge_array = graph_.edge_array_;
         float* const restrict edge_weight_array = graph_.edge_weight_array_;
         const int lgl = graph_.local_bits_;
         const uint32_t local_mask = (uint32_t(1) << lgl) - 1;
         const int64_t L = graph_.num_local_verts_;
         const int r_bits = graph_.r_bits_;
         const uint32_t row_mask = (uint32_t(1) << r_bits) - 1;

         {
            const TwodVertex* const restrict cq_list = sssp_.cq_any_;
            assert(sssp_.cq_size_ % 2 == 0);
#pragma omp for
            for(int64_t i = 0; i < int64_t(sssp_.cq_size_); i += 2) {
               const SeparatedId src(cq_list[i]);
               const TwodVertex src_c = src.value >> lgl;
               const TwodVertex compact = src_c * L + (src.value & local_mask);
               const TwodVertex word_idx = compact >> SsspBase::LOG_NBPE;
               const int bit_idx = compact & SsspBase::NBPE_MASK;
               const BitmapType row_bitmap_i = graph_.row_bitmap_[word_idx];
               const BitmapType mask = BitmapType(1) << bit_idx;

               if(row_bitmap_i & mask) {
                  const BitmapType low_mask = (BitmapType(1) << bit_idx) - 1;
                  const TwodVertex non_zero_off = graph_.row_sums_[word_idx] + __builtin_popcountl(graph_.row_bitmap_[word_idx] & low_mask);
                  const int64_t e_start = graph_.row_starts_[non_zero_off];
                  const int64_t e_end = graph_.row_starts_[non_zero_off + 1];
                  const int64_t root = cq_list[i + 1];

                  for( int64_t e = e_start; e < e_end; ++e ) {
                     const int64_t tgt = edge_array[e];
                     const uint32_t head_unordered = tgt >> (lgl + graph_.r_bits_);
                     const uint32_t row = (tgt >> lgl) & row_mask;
                     const uint32_t owner = (mpi.size_2dr * graph_.edge_head_ownerc_[e] + row);
                     const int64_t head_global = int64_t(mpi.size_2d) * int64_t(head_unordered) + int64_t(owner);

                     if( head_global == root ) {
                        if( !sssp_.top_down_target_is_settled(tgt, r_bits, lgl, L) ) {
                                      std::cout << "fail for  rank=" <<  mpi.rank_2d  << " head=" << head_unordered <<  " edge_weight=" << graph_.edge_weight_array_[e] << " owner=" << owner << " head_global=" << head_global << " tgt=" << graph_.edge_array_[e] << " pos=" << e <<   '\n';
                                  //    assert(0);
                                   }

                       // std::cout << mpi.rank_2d << " for local: " << (src.value & local_mask)  << "->" << head_unordered << " delete: " << (src.value & local_mask) << "->" << (head_global) << " c=" << edge_weight_array[e] << '\n';
                        edge_weight_array[e] = deletion_weight;
#pragma omp atomic update
                        nkilled++;
                     }

                  }
               } // if(row_bitmap_i & mask) {
            } // #pragma omp for // implicit barrier
         }
      }

      int64_t send = nkilled;
      MPI_Reduce(&send, &nkilled, 1, MpiTypeOf<int64_t>::type, MPI_SUM, 0, mpi.comm_2d);
      if( mpi.isMaster() )
         std::cout << "REAL nkilled=" << nkilled << '\n';
   }

   SsspBase& sssp_;
   Graph& graph_;
   LocalVertex root_local_; // current local root for preprocessing
   bool has_root_local_;
   uint32_t root_round_;
   int current_round_;
   int max_n_rounds_;
   int64_t nkilled_all_;
   int64_t n_all_;
};


// deletes redundant edges
void SsspPresolver::presolve_sssp(int niterations, int64_t* pred, float* dist)
{
   SET_AFFINITY;
   TRACER(run_sssp);
   sssp_.pred_ = pred;
   sssp_.dist_ = dist;
   assert(niterations > 0);
   max_n_rounds_ = niterations;
   nkilled_all_ = n_all_ = 0;
   const int n_repeats = 2; // todo parameter
   int max_seconds = std::numeric_limits<int>::max();
   const double start_time = MPI_Wtime();
   int is_stopped = 0;
   const char* presol_time_char = std::getenv("PRESOL_SECONDS");

   if( presol_time_char )
      max_seconds = std::stoi(presol_time_char);
   assert(max_seconds >= 1);

   MPI_Barrier(mpi.comm_2d);

   for( int i = 0; i < n_repeats && !is_stopped; i++ ) {
      root_round_ = i * mpi.rank_2d;
      for( int iter = 0; iter < niterations; iter++ ) {
         current_round_ = iter;
         presolve_round_run();

         int is_stopped_mine = 0;
         if( int((MPI_Wtime() - start_time)) > max_seconds ) is_stopped_mine = 1;

         MPI_Allreduce(&is_stopped_mine, &is_stopped, 1, MPI_INT, MPI_MAX, mpi.comm_2d);

         if( is_stopped ) {
            break;
         }
      }
   }

   if( mpi.isMaster() && is_stopped )
      std::cout << "stopped early!" << '\n';

   if( mpi.isMaster() && n_all_ > 0 )
           std::cout << "FINAL: n_all=" << n_all_ << " n_killed="  << nkilled_all_ << " ratio=" << double(nkilled_all_) / n_all_ << '\n';

   assert(graph_.edge_head_ownerc_);
   free(graph_.edge_head_ownerc_); graph_.edge_head_ownerc_ = nullptr;
}


#endif /* SRC_SSSP_SSSP_PRESOL_HPP_ */

/*
 * sssp.hpp
 *
 *  Created on: Mar 8, 2022
 *      Author: Daniel Rehfeldt
 */


/*
 * bfs.hpp
 *
 *  Created on: Mar 5, 2012
 *      Author: koji
 */

#ifndef SSSP_SRC_SSSP_HPP_
#define SSSP_SRC_SSSP_HPP_

#include <pthread.h>
#include <deque>
#include <limits>
#include <iostream>
#include "utils.hpp"
#include "fiber.hpp"
#include "abstract_comm.hpp"
#include "mpi_comm.hpp"
#include "fjmpi_comm.hpp"
#include "bottom_up_comm.hpp"
#include "sssp_state.hpp"
#include "utils.hpp"
#include "low_level_func.h"

#if VERBOSE_MODE
#include "profiling.hpp"
#endif

#if PROFILING_MODE
   #define profiling_commit(my_commit_) , my_commit_
#else
   #define profiling_commit(my_commit_)
#endif


#define debug(...) debug_print(BFSMN, __VA_ARGS__)

class SsspPresolver;

class SsspBase
{
	typedef SsspBase ThisType;
	typedef Graph2DCSR GraphType;
	static constexpr float delta_step_default = 0.02;
	enum class Work_buf_state { none, cq, dists, settled };
	friend SsspPresolver;
public:
	enum {
		// Number of CQ bitmap entries represent as 1 bit in summary.
		// Since the type of bitmap entry is int32_t and 1 cache line is composed of 32 bitmap entries,
		// 32 is effective value.
		ENABLE_WRITING_DEPTH = 1,

		BUCKET_UNIT_SIZE = 1024,

		// non-parameters
		NBPE = PRM::NBPE,
		LOG_NBPE = PRM::LOG_NBPE,
		NBPE_MASK = PRM::NBPE_MASK,

		BFELL_SORT = PRM::BFELL_SORT,
		LOG_BFELL_SORT = PRM::LOG_BFELL_SORT,
		BFELL_SORT_MASK = PRM::BFELL_SORT_MASK,
		BFELL_SORT_IN_BMP = BFELL_SORT / NBPE,

		BU_SUBSTEP = PRM::NUM_BOTTOM_UP_STREAMS,
	};

	class QueuedVertexes {
	public:
      int64_t preds[BUCKET_UNIT_SIZE];
      LocalVertex v[BUCKET_UNIT_SIZE];
      float weights[BUCKET_UNIT_SIZE];
		int length;
		enum { SIZE = BUCKET_UNIT_SIZE };

		QueuedVertexes() : length(0) { }
		void append_nocheck(LocalVertex val, int64_t pred, float weight) {
		   assert(weight >= 0.0);
		   assert(length < SIZE);
			v[length] = val;
         preds[length] = pred;
			weights[length++] = weight;
		}
		bool full() { return (length == SIZE); }
		int size() { return length; }
		void clear() { length = 0; }
	};

	struct ThreadLocalBuffer {
		QueuedVertexes* cur_buffer;
		LocalPacket fold_packet[1];
	};

	SsspBase()
		: bottom_up_substep_(NULL)
		, top_down_comm_(this)
#ifdef USE_BOTTOM_UP
	   , bottom_up_comm_(this)
#endif
		, td_comm_(mpi.comm_2dc, &top_down_comm_)
#ifdef USE_BOTTOM_UP
		, bu_comm_(mpi.comm_2dr, &bottom_up_comm_)
#endif
		, denom_to_bottom_up_(DENOM_TOPDOWN_TO_BOTTOMUP)
		, denom_bitmap_to_list_(DENOM_BITMAP_TO_LIST)
      , vertices_pos_(NULL)
		, thread_sync_(omp_get_max_threads())
	{
	   const char* delta_step_char = std::getenv("DELTA_STEP");
	   global_visited_vertices_ = 0;
	   nq_buf_length_ = cq_distance_buf_length_ = -1;
	   is_bellman_ford_ = false;
	   is_presolve_mode_ = false;
	   work_buf_state_= Work_buf_state::none;

	   if( delta_step_char ) {
	      delta_step_ = atof(delta_step_char);
	   }
	   else {
	      delta_step_ = delta_step_default;
	   }
	   assert(0.0 < delta_step_ && delta_step_ <= 1.0);

	   if( mpi.isMaster() ) print_with_prefix("delta_step=%f \n", delta_step_);
	}

	virtual ~SsspBase()
	{
	   assert(!vertices_pos_);
		delete bottom_up_substep_; bottom_up_substep_ = NULL;
	}

	template <typename EdgeList>
	void construct(EdgeList* edge_list)
	{
		// minimun requirement of CQ
		// CPU: MINIMUN_SIZE_OF_CQ_BITMAP words -> MINIMUN_SIZE_OF_CQ_BITMAP * NUMBER_PACKING_EDGE_LISTS * mpi.size_2dc
		// GPU: THREADS_PER_BLOCK words -> THREADS_PER_BLOCK * NUMBER_PACKING_EDGE_LISTS * mpi.size_2dc

		int log_local_verts_unit = get_msb_index(std::max<int>(BFELL_SORT, NBPE) * 8);

		detail::GraphConstructor2DCSR<EdgeList> constructor;
		constructor.construct(edge_list, log_local_verts_unit, graph_);
		graph_.separateHeavyEdges(delta_step_);
	}

	void prepare_sssp() {
		printInformation();
		allocate_memory();
	}

	void run_sssp(int64_t root, int64_t* pred, float* dist);

	void end_sssp() {
		deallocate_memory();
	}

   SsspState get_state() {
      const float bucket_upper = (delta_epoch_ + 1.0) * delta_step_;
	   SsspState state = (SsspState){ vertices_isSettled_, bucket_upper, is_bellman_ford_, is_light_phase_, has_settled_vertices_, is_presolve_mode_};
	   return state;
	}

	GraphType graph_;

private:

	int64_t get_bitmap_size_src() const {
		return graph_.num_local_verts_ / NBPE * mpi.size_2dr;
	}
	int64_t get_bitmap_size_tgt() const {
		return graph_.num_local_verts_ / NBPE * mpi.size_2dc;
	}
	int64_t get_bitmap_size_local() const {
		return graph_.num_local_verts_ / NBPE;
	}

	template <typename T>
	void get_shared_mem_pointer(void*& ptr, int64_t width, T** local_ptr, T** orig_ptr) {
		if(orig_ptr) *orig_ptr = (T*)ptr;
		if(local_ptr) *local_ptr = (T*)ptr + width*mpi.rank_z;
		ptr = (uint8_t*)ptr + width*sizeof(T)*mpi.size_z;
	}

	void allocate_memory()
	{
		const int max_threads = omp_get_max_threads();
		const int max_comm_size = std::max(mpi.size_2dc, mpi.size_2dr);
		const int64_t bitmap_width = get_bitmap_size_local();
		pred_presol_ = nullptr;
		dist_presol_ = nullptr;

#if USE_PROPER_HASHMAP
		const int64_t vertices_pos_length = graph_.num_local_verts_;
#else
		const int64_t vertices_pos_length = graph_.num_local_verts_ * max_threads;
#endif

		assert(!vertices_pos_);
		vertices_pos_ = (int32_t*)cache_aligned_xmalloc(vertices_pos_length * sizeof(vertices_pos_[0]));

#pragma omp parallel for schedule(static)
		for( int64_t i = 0; i < vertices_pos_length; ++i )
		   vertices_pos_[i] = -1;

#if USE_DISTANCE_LOCKS
		vertices_locks_ = (omp_lock_t*)cache_aligned_xmalloc(graph_.num_local_verts_ * sizeof(vertices_locks_[0]));

#pragma omp parallel for schedule(static)
		for( int64_t i = 0; i < graph_.num_local_verts_; ++i )
		    omp_init_lock(&vertices_locks_[i]);
#endif


		/**
		 * Buffers for computing BFS
		 * - next queue: This is vertex list in the top-down phase, and <pred, target> tuple list in the bottom-up phase.
		 * - thread local buffer (includes local packet)
		 * - two visited memory (for double buffering)
		 * - working memory: This is used
		 * 	1) to store steaming visited in the bottom-up search phase
		 * 		required size: half_bitmap_width * BOTTOM_UP_BUFFER (for each process)
		 * 	2) to store next queue vertices in the bottom-up expand phase
		 * 		required size: half_bitmap_width * 2 (for each process)
		 * - shared visited:
		 * - shared visited update (to store the information to update shared visited)
		 * - current queue extra memory (is allocated dynamically when the it is required)
		 * - communication buffer for asynchronous communication:
		 */

		a2a_comm_buf_.allocate_memory(graph_.num_local_verts_ * sizeof(int32_t) * 50); // TODO: accuracy previous value: 50

		top_down_comm_.max_num_rows = graph_.num_local_verts_ * 16 / PRM::TOP_DOWN_PENDING_WIDTH + 1000;
		top_down_comm_.tmp_rows = (TopDownRow*)cache_aligned_xmalloc(
				top_down_comm_.max_num_rows*2*sizeof(TopDownRow)); // for debug

		thread_local_buffer_ = (ThreadLocalBuffer**)cache_aligned_xmalloc(sizeof(thread_local_buffer_[0])*max_threads);

		const int bottom_up_vertex_count_per_thread = (bitmap_width/BU_SUBSTEP + max_threads - 1) / max_threads * NBPE;
		const int packet_buffer_length = std::max(
				sizeof(LocalPacket) * max_comm_size, // for the top-down and bottom-up list search
				sizeof(TwodVertex) * 2 * bottom_up_vertex_count_per_thread);
		const int buffer_width = roundup<int>(
				sizeof(ThreadLocalBuffer) + packet_buffer_length, CACHE_LINE);
		buffer_.thread_local_ = cache_aligned_xcalloc(buffer_width*max_threads);
		for(int i = 0; i < max_threads; ++i) {
			ThreadLocalBuffer* tlb = (ThreadLocalBuffer*)
							((uint8_t*)buffer_.thread_local_ + buffer_width*i);
			tlb->cur_buffer = NULL;
			thread_local_buffer_[i] = tlb;
		}
		packet_buffer_is_dirty_ = true;

		enum { NBUF = PRM::BOTTOM_UP_BUFFER };
		work_buf_size_ = std::max<uint64_t>( {
		   // !!!NOTE!!!: for bottom-up we would need max_comm_size below! Currently not used
				bitmap_width * sizeof(BitmapType) * (int64_t) mpi.size_2dc / mpi.size_z, // space to receive NQ bitmap
				bitmap_width * NBUF * sizeof(BitmapType), // space for working buffer
				graph_.num_orig_local_verts_ * sizeof(TwodVertex) // size for copying predecessors
		});

		cq_distance_buf_length_ = work_buf_size_ / sizeof(BitmapType);

		const int64_t total_size_of_shared_memory = -1;
		s_.sync = nullptr;
		s_.offset = nullptr;
		nq_recv_buf_ = nullptr;
		shared_visited_ = nullptr;
		new_visited_ = old_visited_ = visited_buffer_ = visited_buffer_orig_ =  buffer_.shared_memory_ = nullptr;
		work_buf_ = (int8_t*) page_aligned_xcalloc(work_buf_size_);

#if 0
		const int shared_offset_length = (max_threads * mpi.size_z * BU_SUBSTEP + 1);
		const int64_t total_size_of_shared_memory =
				bitmap_width * 3 * sizeof(BitmapType) * mpi.size_z + // new and old visited and buffer
				work_buf_size_ * mpi.size_z + // work_buf_
				bitmap_width * sizeof(BitmapType) * max_comm_size + // shared visited memory
				sizeof(memory::SpinBarrier) + sizeof(int) * shared_offset_length;
		VERBOSE(if(mpi.isMaster()) print_with_prefix("Allocating shared memory: %f GB per node.", to_giga(total_size_of_shared_memory)));

		void* smem_ptr = buffer_.shared_memory_ = shared_malloc(total_size_of_shared_memory);

		get_shared_mem_pointer<BitmapType>(smem_ptr, bitmap_width, (BitmapType**)&new_visited_, NULL);
		get_shared_mem_pointer<BitmapType>(smem_ptr, bitmap_width, (BitmapType**)&old_visited_, NULL);
		get_shared_mem_pointer<BitmapType>(smem_ptr, bitmap_width, (BitmapType**)&visited_buffer_, (BitmapType**)&visited_buffer_orig_);
		get_shared_mem_pointer<int8_t>(smem_ptr, work_buf_size_, (int8_t**)&work_buf_, (int8_t**)&nq_recv_buf_);

		shared_visited_ = (BitmapType*)smem_ptr;
		smem_ptr = (BitmapType*)smem_ptr + bitmap_width * max_comm_size;
		s_.sync = new (smem_ptr) memory::SpinBarrier(mpi.size_z);
		smem_ptr = s_.sync + 1;
		s_.offset = (int*)smem_ptr;
		smem_ptr = (int*)smem_ptr + shared_offset_length;

		assert(smem_ptr == (int8_t*)buffer_.shared_memory_ + total_size_of_shared_memory);
#endif

		//vertices_isInCurrentBucket_ = (BitmapType*)cache_aligned_xmalloc(bitmap_width * sizeof(*vertices_isInCurrentBucket_));

		bottom_up_substep_ = new MpiBottomUpSubstepComm(mpi.comm_2dr);
		bottom_up_substep_->register_memory(buffer_.shared_memory_, total_size_of_shared_memory);

		nq_buf_length_ = bitmap_width;
		cq_root_list_ = nullptr;
		nq_root_list_ = nullptr;
		cq_distance_list_ = (float*)page_aligned_xmalloc(cq_distance_buf_length_ * sizeof(*cq_distance_list_));
		nq_distance_list_ = (float*)page_aligned_xmalloc(nq_buf_length_ * sizeof(*nq_distance_list_));
		nq_list_ = (TwodVertex*)page_aligned_xmalloc(nq_buf_length_ * sizeof(*nq_list_));

		assert(graph_.num_local_verts_ % NBPE == 0);
		vertices_isSettled_ = (BitmapType*)cache_aligned_xmalloc(bitmap_width * sizeof(*vertices_isSettled_) * mpi.size_2dr);
		vertices_isSettledLocal_ = (BitmapType*)cache_aligned_xmalloc(bitmap_width * sizeof(*vertices_isSettledLocal_));

		cq_any_ = NULL;
		global_nq_size_ = max_nq_size_ = nq_size_ = cq_size_ = 0;
		bitmap_or_list_ = false;
	}

	void deallocate_memory()
	{
	   assert(!cq_root_list_ && !nq_root_list_);
	   free(vertices_isSettledLocal_); vertices_isSettledLocal_ = NULL;
	   free(vertices_isSettled_); vertices_isSettled_ = NULL;
	   free(cq_distance_list_); cq_distance_list_ = NULL;
	   free(nq_distance_list_); nq_distance_list_ = NULL;
	   free(nq_list_); nq_list_ = NULL;
	   free(vertices_pos_); vertices_pos_ = NULL;

#if USE_DISTANCE_LOCKS
#pragma omp parallel for schedule(static)
      for( int64_t i = 0; i < graph_.num_local_verts_; ++i )
         omp_destroy_lock(&vertices_locks_[i]);
      free(vertices_locks_); vertices_locks_ = NULL;
#endif

		free(buffer_.thread_local_); buffer_.thread_local_ = NULL;
		//shared_free(buffer_.shared_memory_); buffer_.shared_memory_ = NULL;
		free(work_buf_);
		free(thread_local_buffer_); thread_local_buffer_ = NULL;
		a2a_comm_buf_.deallocate_memory();
	}

	// updates capacity of next queue. Only call with one thread!
	void update_nq_capacity(int result_size)
   {
	   assert(nq_buf_length_ >= get_bitmap_size_local());
      if( result_size > nq_buf_length_ ) {
         nq_buf_length_ = result_size;
         free(nq_list_);
         free(nq_distance_list_);
         nq_distance_list_ = (float*) page_aligned_xmalloc(nq_buf_length_ * sizeof(*nq_distance_list_));
         nq_list_ = (TwodVertex*) page_aligned_xmalloc(nq_buf_length_ * sizeof(*nq_list_));
         // std::cout << mpi.rank_2d << " reallocate nq for bucket generation!" << '\n';

         if( nq_root_list_ ) {
            free(nq_root_list_);
            nq_root_list_ = (int64_t*) page_aligned_xmalloc(nq_buf_length_ * sizeof(*nq_root_list_));
         }
      }
   }

   // updates work-buffer. Only call with one thread!
   void update_work_buf(int64_t new_size) {
      assert(omp_get_thread_num() == 0);
      assert(new_size % sizeof(int64_t) == 0);
      if( work_buf_size_ < new_size ) {
         work_buf_size_ = new_size;
         free(work_buf_);
         work_buf_ = page_aligned_xmalloc(work_buf_size_);
      }
   }

public:
	//-------------------------------------------------------------//
	// Async communication
	//-------------------------------------------------------------//

	class CommBufferPool {
	public:
		void allocate_memory(int size) {
			first_buffer_ = cache_aligned_xmalloc(size);
			second_buffer_ = cache_aligned_xmalloc(size);
			current_index_ = 0;
			pool_buffer_size_ = size;
			num_buffers_ = size / PRM::COMM_BUFFER_SIZE;
		}

		void deallocate_memory() {
			free(first_buffer_); first_buffer_ = NULL;
			free(second_buffer_); second_buffer_ = NULL;
		}

		void* get_next() {
			int idx = current_index_++;
			if(num_buffers_ <= idx) {
				fprintf(IMD_OUT, "num_buffers_ <= idx (num_buffers=%d)\n", num_buffers_);
				throw "Error: buffer size not enough";
			}
			return (uint8_t*)first_buffer_ + PRM::COMM_BUFFER_SIZE * idx;
		}

		void* clear_buffers() {
			current_index_ = 0;
			return first_buffer_;
		}

		void* second_buffer() {
			return second_buffer_;
		}

		int pool_buffer_size() {
			return pool_buffer_size_;
		}

	protected:
		int pool_buffer_size_;
		void* first_buffer_;
		void* second_buffer_;
		int current_index_;
		int num_buffers_;
	};

	template <typename T>
	class CommHandlerBase : public AlltoallBufferHandler {
	public:
		enum { BUF_SIZE = PRM::COMM_BUFFER_SIZE / sizeof(T) };
		CommHandlerBase(ThisType* this__)
			: this_(this__)
			, pool_(&this__->a2a_comm_buf_)
		{ }
		virtual ~CommHandlerBase() { }
		virtual void* get_buffer() {
			return this->pool_->get_next();
		}
		virtual void add(void* buffer, void* ptr__, int offset, int length) {
			assert (offset >= 0);
			assert (offset + length <= BUF_SIZE);
			memcpy((T*)buffer + offset, ptr__, length*sizeof(T));
		}
		virtual void* clear_buffers() {
			return this->pool_->clear_buffers();
		}
		virtual void* second_buffer() {
			return this->pool_->second_buffer();
		}
		virtual int max_size() {
			return this->pool_->pool_buffer_size();
		}
		virtual int buffer_length() {
			return BUF_SIZE;
		}
		virtual MPI_Datatype data_type() {
			return MpiTypeOf<T>::type;
		}
		virtual int element_size() {
			return sizeof(T);
		}
		virtual void finish() { }
	protected:
		ThisType* this_;
		CommBufferPool* pool_;
	};

	struct TopDownRow {
		int64_t src;
		int length;
		uint32_t* ptr;
	};

	class TopDownCommHandler : public CommHandlerBase<uint32_t> {
	public:
		TopDownCommHandler(ThisType* this__)
			: CommHandlerBase<uint32_t>(this__)
			, tmp_rows(NULL)
			, max_num_rows(0)
			, num_rows(0)
			  { }

		~TopDownCommHandler() {
			if(tmp_rows != NULL) { free(tmp_rows); tmp_rows = NULL; }
		}

      virtual void received(void *buf, int offset, int length, int src, bool is_ptr)
      {
         VERBOSE(g_tp_comm += length * sizeof(uint32_t));

         if( length == 0 )
            return;

         const int thread_id = omp_get_thread_num();

         if( is_ptr ) {
            if( this_->is_light_phase_) {
               this->this_->top_down_receive_ptr<true>((uint32_t*) buf + offset, length, tmp_rows, &num_rows, thread_id);
            }
            else {
               assert(!this_->is_bellman_ford_);
               this->this_->top_down_receive_ptr<false>((uint32_t*) buf + offset, length, tmp_rows, &num_rows, thread_id);
            }
         }
         else {
            if( this_->is_light_phase_ ) {
               this->this_->top_down_receive<true>((uint32_t*) buf + offset, length, thread_id);
            }
            else {
               assert(!this_->is_bellman_ford_);
               this->this_->top_down_receive<false>((uint32_t*) buf + offset, length, thread_id);
            }
         }

         assert(num_rows < max_num_rows);
      }

		virtual void finish() {
			//VERBOSE(if(mpi.isMaster()) print_with_prefix("num_rows= %d / %d", num_rows, max_num_rows));
			if(num_rows == 0) return ;
			if(num_rows > max_num_rows) {
				fprintf(IMD_OUT, "Insufficient temporary rows buffer\n");
				throw "Insufficient temporary rows buffer";
			}

			if(this_->is_light_phase_) {
				this->this_->top_down_row_receive<true>(tmp_rows, num_rows);
			}
			else {
			   assert(!this_->is_bellman_ford_);
				this->this_->top_down_row_receive<false>(tmp_rows, num_rows);
			}
			num_rows = 0;
		}

		TopDownRow* tmp_rows;
		int max_num_rows;
		volatile int num_rows;
	};

#ifdef USE_BOTTOM_UP
	class BottomUpCommHandler : public CommHandlerBase<int64_t> {
	public:
		BottomUpCommHandler(ThisType* this__)
			: CommHandlerBase<int64_t>(this__)
			  { }

		virtual void received(void* buf, int offset, int length, int src) {
			VERBOSE(g_bu_pred_comm += length * sizeof(int64_t));
			BottomUpReceiver recv(this->this_, (int64_t*)buf + offset, length, src);
			recv.run();
		}
	};

	//-------------------------------------------------------------//
	// expand phase
	//-------------------------------------------------------------//

	template <typename T>
	void get_visited_pointers(T** ptrs, int num_ptrs, void* visited_buf, int split_count) {
		int step_bitmap_width = get_bitmap_size_local() / split_count;
		for(int i = 0; i < num_ptrs; ++i) {
			ptrs[i] = (T*)((BitmapType*)visited_buf + step_bitmap_width*i);
		}
	}
#endif


	void clear_nq_stack() {
		int num_buffers = nq_.stack_.size();
		for(int i = 0; i < num_buffers; ++i) {
			// Since there are no need to lock pool in this case,
			// we invoke Pool::free method explicitly.
			nq_.stack_[i]->length = 0;
			nq_empty_buffer_.memory::Pool<QueuedVertexes>::free(nq_.stack_[i]);
		}
		nq_.stack_.clear();
	}

	void first_expand(int64_t root) {
#if VERBOSE_MODE
	   const double expand_start_time = MPI_Wtime();
#endif

		// !!root is UNSWIZZLED and ORIGINAL ID!!
      const int root_owner = vertex_owner(root);
      const int root_r = root_owner % mpi.size_2dr; // the row in which the owner lives
      const int root_c = root_owner / mpi.size_2dr; // the column in which the owner lives

		cq_any_ = (TwodVertex*)work_buf_;
		work_buf_state_ = Work_buf_state::cq;
		assert(cq_distance_list_);
		reset_root_grad1 = false;

		if(root_r == mpi.rank_2dr) {
			// get reordered vertex id and send it to all processes in this row
			int64_t root_reordered = 0;
			if(root_owner == mpi.rank_2d) {
				// yes, we have reordered id
				const int64_t root_local = vertex_local(root);
				const int64_t reordered = graph_.reorder_map_[root_local];
				root_reordered = (reordered * mpi.size_2d) + root_owner;

				MPI_Bcast(&root_reordered, 1, MpiTypeOf<int64_t>::type, root_c, mpi.comm_2dr);

				pred_[reordered] = root;
				dist_[reordered] = 0.0;

				if( graph_.local_vertex_isDeg1(reordered) ) {
				   const int64_t word_idx = reordered >> LOG_NBPE;
				   const int64_t bit_idx = reordered & NBPE_MASK;
				   graph_.is_grad1_bitmap_[word_idx] ^= BitmapType(1) << bit_idx;
				   assert(!graph_.local_vertex_isDeg1(reordered));
				   reset_root_grad1 = true;
				}

			}
			else {
				MPI_Bcast(&root_reordered, 1, MpiTypeOf<int64_t>::type, root_c, mpi.comm_2dr);
			}

			// update CQ
			const SeparatedId root_src = graph_.VtoS(root_reordered);
			cq_any_[0] = root_src.value;
			cq_distance_list_[0] = 0.0;
			if( cq_root_list_ ) cq_root_list_[0] = root;
			cq_size_ = 1;
		}
		else {
			cq_size_ = 0;
		}

#if VERBOSE_MODE
      assert((MPI_Wtime() - expand_start_time) >= 0.0);
      profiling::expand_time += (MPI_Wtime() - expand_start_time);
#endif
	}

   // expands settled vertices bitmap
	// todo avoid code duplication; if expand_visited_bitmap is kept, merge the two
   void expand_vertices_bitmap(BitmapType* bitmap_local, BitmapType* bitmap_global) {
      TRACER(expand_vis_bmp);
      const int bitmap_width = get_bitmap_size_local();
      assert(!mpi.isYdimAvailable());
      assert(mpi.rank_z == 0 && mpi.comm_y != MPI_COMM_NULL);
      assert(mpi.comm_c.size == mpi.size_2dr);

      BitmapType* send_buffer = bitmap_local;
      BitmapType* recv_buffer = bitmap_global;

#if ENABLE_MY_ALLGATHER == 1
      assert(!mpi.isYdimAvailable());
      MpiCol::my_allgather(send_buffer, bitmap_width, recv_buffer, mpi.comm_c);
#elif ENABLE_MY_ALLGATHER == 2
      assert(!mpi.isYdimAvailable());
      my_allgather_2d(send_buffer, bitmap_width, recv_buffer, mpi.comm_c);
#else
      MPI_Allgather(send_buffer, bitmap_width, get_mpi_type(send_buffer[0]), recv_buffer, bitmap_width, get_mpi_type(recv_buffer[0]), mpi.comm_2dc);
#endif

#if VERBOSE_MODE
      g_expand_bitmap_comm += bitmap_width * mpi.size_y * sizeof(BitmapType);
#endif
   }

#ifdef USE_BOTTOM_UP
	// expand visited bitmap and receive the shared visited
	void expand_visited_bitmap() {
		TRACER(expand_vis_bmp);
		int bitmap_width = get_bitmap_size_local();
		if(mpi.isYdimAvailable()) s_.sync->barrier();
		if(mpi.rank_z == 0 && mpi.comm_y != MPI_COMM_NULL) {
	      BitmapType* const bitmap = (BitmapType*)new_visited_;
	      BitmapType* recv_buffer = shared_visited_;
			// TODO: asymmetric size for z. (MPI_Allgather -> MPI_Allgatherv or MpiCol::allgatherv ?)
			int shared_bitmap_width = bitmap_width * mpi.size_z;
#if ENABLE_MY_ALLGATHER
			if(mpi.isYdimAvailable()) {
				if(mpi.isMaster()) print_with_prefix("Error: MY_ALLGATHER does not support shared memory Y dimension.");
			}
#if ENABLE_MY_ALLGATHER == 1
		MpiCol::my_allgather(bitmap, shared_bitmap_width, recv_buffer, mpi.comm_c);
#else
		my_allgather_2d(bitmap, shared_bitmap_width, recv_buffer, mpi.comm_c);
#endif // #if ENABLE_MY_ALLGATHER == 1
#else
			MPI_Allgather(bitmap, shared_bitmap_width, get_mpi_type(bitmap[0]),
					recv_buffer, shared_bitmap_width, get_mpi_type(bitmap[0]), mpi.comm_y);
#endif
#if VERBOSE_MODE
			g_expand_bitmap_comm += shared_bitmap_width * mpi.size_y * sizeof(BitmapType);
#endif
		}
		if(mpi.isYdimAvailable()) s_.sync->barrier();
	}

	// expand visited bitmap and receive the current queue
	void expand_nq_bitmap() {
		TRACER(expand_vis_bmp);
		int bitmap_width = get_bitmap_size_local();
		BitmapType* const bitmap = (BitmapType*)new_visited_;
		BitmapType* recv_buffer = shared_visited_;
#if ENABLE_MY_ALLGATHER
		if(mpi.isYdimAvailable()) {
			if(mpi.isMaster()) print_with_prefix("Error: MY_ALLGATHER does not support shared memory Y dimension.");
		}
#if ENABLE_MY_ALLGATHER == 1
	MpiCol::my_allgather(bitmap, bitmap_width, recv_buffer, mpi.comm_r);
#else
	my_allgather_2d(bitmap, bitmap_width, recv_buffer, mpi.comm_r);
#endif // #if ENABLE_MY_ALLGATHER == 1
#else
		MPI_Allgather(bitmap, bitmap_width, get_mpi_type(bitmap[0]),
				recv_buffer, bitmap_width, get_mpi_type(bitmap[0]), mpi.comm_2dr);
#endif
#if VERBOSE_MODE
		g_expand_bitmap_comm += bitmap_width * mpi.size_2dc * sizeof(BitmapType);
#endif
	}

	int expand_visited_list(int node_nq_size) {
		TRACER(expand_vis_list);
		if(mpi.rank_z == 0 && mpi.comm_y != MPI_COMM_NULL) {
			s_.offset[0] = MpiCol::allgatherv((TwodVertex*)visited_buffer_orig_,
					 nq_recv_buf_, node_nq_size, mpi.comm_y, mpi.size_y);
			VERBOSE(g_expand_list_comm += s_.offset[0] * sizeof(TwodVertex));
		}
		if(mpi.isYdimAvailable()) s_.sync->barrier();
		return s_.offset[0];
	}
#endif

   // marks already settled vertices in new_visited bitmap
   void bucket_mark_settled_bitmap() {
      const uint64_t num_local_verts = graph_.num_local_verts_;
      const float bbound_upper = (delta_epoch_ + 1) * delta_step_;
      const float bbound_lower = (delta_epoch_) * delta_step_;
      BitmapType* const restrict is_settled = (BitmapType*)vertices_isSettledLocal_;

#pragma omp parallel for schedule(static)
      for( uint64_t i = 0; i < num_local_verts; i++ ) {

#if 0
         if( dist_[i] > bbound_upper && dist_[i] < bbound_lower + graph_.vertices_minweight_[i]  )
            printf("yeaB %d \n", i);
#endif

         if( dist_[i] < bbound_upper || dist_[i] < bbound_lower + graph_.vertices_minweight_[i] ) {
            const uint64_t word = i >> LOG_NBPE;
            const uint64_t bit = i & NBPE_MASK;
            assert(word == i / NBPE && bit == i % NBPE);

            is_settled[word] |= BitmapType(1) << bit;
         }
      }
   }

   // expands settled vertices nq list (to bitmap)
   void expand_settled_list(int n_settled) {
      const int comm_size = mpi.comm_c.size;
      int recv_size[comm_size];
      int recv_off[comm_size + 1];
      MPI_Allgather(&n_settled, 1, MPI_INT, recv_size, 1, MPI_INT, mpi.comm_c.comm);
      recv_off[0] = 0;
      for(int i = 0; i < comm_size; ++i)
         recv_off[i + 1] = recv_off[i] + recv_size[i];
      const int settled_size = recv_off[comm_size];

      update_work_buf(settled_size * int64_t(sizeof(TwodVertex)));
      TwodVertex* recv_buf = (TwodVertex*) work_buf_;

#if ENABLE_MY_ALLGATHER == 1
      MpiCol::my_allgatherv(nq_list_, n_settled, recv_buf, recv_size, recv_off, mpi.comm_c);
#elif ENABLE_MY_ALLGATHER == 2
      my_allgatherv_2d(nq_list_, n_settled, recv_buf, recv_size, recv_off, mpi.comm_c);
#else
      MPI_Allgatherv(nq_list_, n_settled, MpiTypeOf<TwodVertex>::type, recv_buf, recv_size, recv_off, MpiTypeOf<TwodVertex>::type, mpi.comm_c.comm);
#endif

      const int lgl = graph_.local_bits_;
      const uint32_t local_mask = (uint32_t(1) << lgl) - 1;
      const int64_t num_local_verts = graph_.num_local_verts_;

      if( !settled_is_clean )
         memory::clean_mt(vertices_isSettled_, get_bitmap_size_local() * mpi.size_2dr * sizeof(*vertices_isSettled_));

#pragma omp parallel for schedule(static)
      for( int i = 0; i < settled_size; i++ ) {
         const SeparatedId src(recv_buf[i]);
         const TwodVertex src_c = src.value >> lgl;
         const TwodVertex compact = src_c * num_local_verts + (src.value & local_mask);
         const TwodVertex word_idx = compact >> LOG_NBPE;
         const int bit_idx = compact & NBPE_MASK;

         assert(word_idx < uint64_t(get_bitmap_size_local() * mpi.size_2dr));
#pragma omp atomic update
         vertices_isSettled_[word_idx] |= (uint64_t(1) << bit_idx);
      }

      assert(!mpi.isYdimAvailable());
   }

   // marks newly settled vertices and stores in nq_list (but also marks bitmap)
   void bucket_mark_settled_list(int& n_settled_new) {
      const uint64_t num_local_verts = graph_.num_local_verts_;
      const float bbound_upper = (delta_epoch_ + 1) * delta_step_;
      const float bbound_lower = (delta_epoch_) * delta_step_;
      BitmapType* const is_settled = (BitmapType*)vertices_isSettledLocal_;
      const int max_threads = omp_get_max_threads();
      int threads_offset[max_threads + 1];
      const TwodVertex shifted_r = TwodVertex(mpi.rank_2dr) << graph_.local_bits_;

#pragma omp parallel
      {
         const int tid = omp_get_thread_num();
         int count = 0;

#pragma omp for schedule(static) nowait
         for( uint64_t i = 0; i < num_local_verts; i++ ) {
            if( dist_[i] < bbound_upper || dist_[i] < bbound_lower + graph_.vertices_minweight_[i] ) {
               const uint64_t word = i >> LOG_NBPE;
               const uint64_t bit = i & NBPE_MASK;
               assert(word == i / NBPE && bit == i % NBPE);

               if( !(is_settled[word] & BitmapType(1) << bit) )
                  count++;
            }
         }
         threads_offset[tid + 1] = count;
#pragma omp barrier
#pragma omp single
         {
            threads_offset[0] = 0;
            for( int i = 1; i < max_threads; i++ )
               threads_offset[i + 1] += threads_offset[i];

            n_settled_new = threads_offset[max_threads];
            update_nq_capacity(n_settled_new);
         } // barrier

         int offset = threads_offset[tid];
#pragma omp for schedule(static) nowait
         for( uint64_t i = 0; i < num_local_verts; i++ ) {
            if( dist_[i] < bbound_upper || dist_[i] < bbound_lower + graph_.vertices_minweight_[i] ) {
               const uint64_t word = i >> LOG_NBPE;
               const uint64_t bit = i & NBPE_MASK;
               assert(word == i / NBPE && bit == i % NBPE);

               if( !(is_settled[word] & BitmapType(1) << bit) ) {
                  is_settled[word] |= BitmapType(1) << bit;
                  nq_list_[offset++] = i | shifted_r;
               }
            }
         }
      } // parallel region
   }

   // expands settled vertices as bitmap
   void bucket_expand_settled_bitmap(int64_t global_nq_size) {
#if VERBOSE_MODE
      const double expand_start_time = MPI_Wtime();
#endif

      const double ratio = double(global_nq_size) / double(graph_.num_global_verts_);

      // todo use parameter
      if( ratio < 0.008 ) {
         int n_settled_new;
         bucket_mark_settled_list(n_settled_new);
         expand_settled_list(n_settled_new);

         if( mpi.isMaster() )
            printf("...expanded settled vertices! (list + calib) \n");
      }
      else {
         bucket_mark_settled_bitmap();
         expand_vertices_bitmap(vertices_isSettledLocal_, vertices_isSettled_);

         if( mpi.isMaster() )
            printf("...expanded settled vertices! (bitmap + calib) \n");
      }

      settled_is_clean = true;
      has_settled_vertices_ = true;

#if VERBOSE_MODE
      assert((MPI_Wtime() - expand_start_time) >= 0.0);
      profiling::expand_settled_bitmap_time += (MPI_Wtime() - expand_start_time);
#endif
   }

	// gets index of next non-empty bucket
   int bucket_get_next_nonempty(bool with_z) {
      TRACER(td_make_nq_list);
      assert(!with_z && "currently not supported"); // if ever use with_z, then change as in top_down_make_nq_list
      assert(!is_bellman_ford_);

      const int max_threads = omp_get_max_threads();
      const float bbound_lower = (delta_epoch_ + 1) * delta_step_;
      float mindists[max_threads];

      //printf("[%f, %f] \n", bbound_lower, bbound_upper);
#pragma omp parallel
      {
         const uint64_t num_local_verts = uint64_t(graph_.num_local_verts_);
         const int tid = omp_get_thread_num();
         float min = std::numeric_limits<float>::max();

#pragma omp for schedule(static) nowait
         for( uint64_t i = 0; i < num_local_verts; i++ ) {
            if( comp::isGE(dist_[i], bbound_lower) && dist_[i] < min )
               min = dist_[i];
         }

         mindists[tid] = min;
      } // omp parallel

      float min = mindists[0];
      for( int i = 1; i < max_threads; i++ )
         if( mindists[i] < min )
            min = mindists[i];

      int index = std::numeric_limits<int>::max();
      if( min < comp::infinity )
      {
         const double delta_step = delta_step_;
         index = int(double(min) / delta_step - double(comp::eps_default));
         if( !comp::isGE(min, index * delta_step) || index == delta_epoch_ )
            index++;

         if( min >= (index + 1.0) * delta_step )
            index++;

         assert(comp::isGE(min, index * delta_step) && min < (index + 1.0) * delta_step);

         // todo remove
         if( !comp::isGE(min, index * delta_step) || min >= (index + 1.0) * delta_step ) {
            printf("numerics issue with delta step! \n");
            print_with_prefix("issue: min=%f (index + 1) * delta_step)=%f  delta_epoch_=%d", min, (index + 1) * delta_step, delta_epoch_);
            print_with_prefix("issue: min=%f index * delta_step_=%f, index=%d", min, index * delta_step, index);
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
         }
      }

      MPI_Allreduce(MPI_IN_PLACE, &index, 1, MpiTypeOf<int>::type, MPI_MIN, mpi.comm_2d);

      return index;
   }

   int64_t bucket_get_nq_size() {
      const float bbound_lower = delta_epoch_ * delta_step_;
      const float bbound_upper = is_bellman_ford_ ? comp::infinity : (delta_epoch_ + 1.0) * delta_step_;

      const uint64_t num_local_verts = uint64_t(graph_.num_local_verts_);
      int count = 0;

#pragma omp parallel for reduction(+: count) schedule(static)
      for( uint64_t i = 0; i < num_local_verts; i++ ) {
         if( comp::isGE(dist_[i], bbound_lower) && dist_[i] < bbound_upper ) {
            if( graph_.local_vertex_isDeg1(i) ) {
               continue;
            }
            count++;
         }
      }

      int64_t send_nq_size = count;
      int64_t nq_sum;
      MPI_Allreduce(&send_nq_size, &nq_sum, 1, MpiTypeOf<int64_t>::type, MPI_SUM, mpi.comm_2d);
      return nq_sum;
   }

   int bucket_make_nq_list(bool with_z, TwodVertex shifted_rc) {
      TRACER(td_make_nq_list);
      assert(!with_z && "currently not supported"); // if ever use with_z, then change as in top_down_make_nq_list

      const int max_threads = omp_get_max_threads();
      int threads_offset[max_threads + 1];
      int result_size = -1;
      const float bbound_lower = delta_epoch_ * delta_step_;
      const float bbound_upper = is_bellman_ford_ ? comp::infinity : (delta_epoch_ + 1.0) * delta_step_;

      //printf("[%f, %f] \n", bbound_lower, bbound_upper);
#pragma omp parallel
      {
         const uint64_t num_local_verts = uint64_t(graph_.num_local_verts_);
         const int tid = omp_get_thread_num();
         int count = 0;

#pragma omp for schedule(static) nowait
         for( uint64_t i = 0; i < num_local_verts; i++ ) {
            if( comp::isGE(dist_[i], bbound_lower) && dist_[i] < bbound_upper ) {
               if( graph_.local_vertex_isDeg1(i) ) {
                  continue;
               }
               count++;
            }
         }
         threads_offset[tid + 1] = count;
#pragma omp barrier
#pragma omp single
         {

            threads_offset[0] = 0;
            for( int i = 0; i < max_threads; i++ )
               threads_offset[i + 1] += threads_offset[i];

            result_size = threads_offset[max_threads];
            update_nq_capacity(result_size);
         } // barrier

#ifndef NDEBUG
#pragma omp for schedule(static)
         for(int i = 0; i < result_size; ++i)
            nq_list_[i] = num_local_verts;
#endif
         int offset = threads_offset[tid];
         const bool next_bitmap_or_list = next_bitmap_or_list_;
#pragma omp for schedule(static) nowait
         for( uint64_t i = 0; i < num_local_verts; i++ ) {
            if( comp::isGE(dist_[i], bbound_lower) && dist_[i] < bbound_upper ) {
               if( graph_.local_vertex_isDeg1(i) ) {
                  continue;
               }
               assert(nq_list_[offset] == num_local_verts);

               if( next_bitmap_or_list )
                  nq_list_[offset] = i;
               else
                  nq_list_[offset] = i | shifted_rc;
               if( is_presolve_mode_ ) nq_root_list_[offset] = pred_[i];
               nq_distance_list_[offset++] = dist_[i];
            }
         }

         // todo remove
         if( offset != threads_offset[tid + 1] ) {
            std::cout << "next bucket build error! \n";
            print_with_prefix("bucket issue!");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
         }
      }

      return result_size;
   }


	int top_down_make_nq(bool with_z, TwodVertex shifted_rc) {
		TRACER(td_make_nq_list);
		const int size_z = with_z ? mpi.size_z : 1;
		const int rank_z = with_z ? mpi.rank_z : 0;
		assert(!with_z && "currently not supported");

		const int max_threads = omp_get_max_threads();
		const int node_threads = max_threads * size_z;

		int th_offset_storage[max_threads+1];
		int *th_offset = with_z ? s_.offset : th_offset_storage;

		int64_t* nq_preds = nullptr;

		int result_size = 0;
		const int num_buffers = nq_.stack_.size();
#pragma omp parallel
		{
			const int tid = omp_get_thread_num() + max_threads * rank_z;
			int count = 0;
#pragma omp for schedule(static) nowait
			for(int i = 0; i < num_buffers; ++i) {
				count += nq_.stack_[i]->length;
			}
			th_offset[tid+1] = count;
#pragma omp barrier
#pragma omp single
			{
            th_offset[0] = 0;
            for(int i = 0; i < node_threads; ++i) {
               th_offset[i+1] += th_offset[i];
            }

				//if(with_z) s_.sync->barrier();
				result_size = th_offset[node_threads];

            update_nq_capacity(result_size);

            if( is_presolve_mode_ )
               nq_preds = nq_root_list_;
            else
               nq_preds = (int64_t*) cache_aligned_xmalloc(result_size * sizeof(*nq_preds));
			} // implicit barrier
#ifndef NDEBUG
#pragma omp for schedule(static)
			for(int i = 0; i < result_size; ++i)
			   nq_list_[i] = graph_.num_local_verts_;
#endif

			int offset = th_offset[tid];
#pragma omp for schedule(static) nowait
			for(int i = 0; i < num_buffers; ++i) {
			   assert(nq_preds);
				const int len = nq_.stack_[i]->length;
				const LocalVertex* const src = nq_.stack_[i]->v;
            const int64_t* const preds = nq_.stack_[i]->preds;
				const float* const src_weights = nq_.stack_[i]->weights;
				TwodVertex* restrict dst = nq_list_ + offset;
				int64_t* restrict dst_preds = nq_preds + offset;
				float* restrict dst_weights = nq_distance_list_ + offset;
				assert(len + offset <= nq_buf_length_);

				for(int c = 0; c < len; ++c) {
					dst[c] = src[c]; // | shifted_rc;
					dst_weights[c] = src_weights[c];
					dst_preds[c] = preds[c];
				}

				offset += len;
			}
			assert (offset == th_offset[tid+1]);
		} // implicit barrier

      const int64_t num_local_verts = graph_.num_local_verts_;

		// filter out duplicates
		for( int i = 0; i < result_size; i++ ) {
		   const int64_t vertex = int64_t(nq_list_[i]);

		   assert(0 <= vertex && vertex < num_local_verts);
		   if( vertices_pos_[vertex] < 0 ) {
		      vertices_pos_[vertex] = i;
		      dist_[vertex] = nq_distance_list_[i];
		      pred_[vertex] = nq_preds[i];
		      continue;
		   }

		   const int twin_pos = vertices_pos_[vertex];
		   assert(nq_list_[i] == nq_list_[twin_pos]);
		   assert(twin_pos < i);

		   if( nq_distance_list_[i] < nq_distance_list_[twin_pos] ) {
            vertices_pos_[vertex] = i;
		      nq_list_[twin_pos] = num_local_verts;

            dist_[vertex] = nq_distance_list_[i];
            pred_[vertex] = nq_preds[i];
		   }
		   else {
		      nq_list_[i] = num_local_verts;
		   }
		}

		const int result_size_old = result_size;
		result_size = 0;

      if( next_bitmap_or_list_ ) {
         for( int i = 0; i < result_size_old; i++ ) {
            const TwodVertex vertex = nq_list_[i];
            if( vertex >= uint64_t(num_local_verts) )
               continue;

            vertices_pos_[vertex] = -1;
            if( graph_.local_vertex_isDeg1(vertex) )
               continue;

            nq_list_[result_size] = vertex; // here is the difference
            if( is_presolve_mode_ ) nq_root_list_[result_size] = nq_root_list_[i];
            nq_distance_list_[result_size++] = nq_distance_list_[i];
         }
         assert(0 && "todo should not be used by now");
      }
      else {
         for( int i = 0; i < result_size_old; i++ ) {
            const TwodVertex vertex = nq_list_[i];
            if( vertex >= uint64_t(num_local_verts) )
               continue;

            vertices_pos_[vertex] = -1;
            if( graph_.local_vertex_isDeg1(vertex) )
               continue;

            nq_list_[result_size] = vertex | shifted_rc; // here is the difference
            if( is_presolve_mode_ ) nq_root_list_[result_size] = nq_root_list_[i];
            nq_distance_list_[result_size++] = nq_distance_list_[i];
         }
      }

      if( !is_presolve_mode_ )
         free(nq_preds);
		return result_size;
	}

	void top_down_expand_nq(int nq_size) {
		TRACER(td_expand_nq_list);
		assert(nq_size >= 0);
		const int comm_size = mpi.comm_r.size;
		int recv_size[comm_size];
		int recv_off[comm_size+1];
		MPI_Allgather(&nq_size, 1, MPI_INT, recv_size, 1, MPI_INT, mpi.comm_r.comm);
		recv_off[0] = 0;
		for(int i = 0; i < comm_size; ++i) {
			recv_off[i+1] = recv_off[i] + recv_size[i];
		}
      cq_size_ = recv_off[comm_size];

		if( cq_distance_buf_length_ < int64_t(cq_size_) ) {
		   cq_distance_buf_length_ = int64_t(cq_size_);
		   free(cq_distance_list_);
		   cq_distance_list_ = (float*)page_aligned_xmalloc(cq_distance_buf_length_ * sizeof(*cq_distance_list_));

		   if( cq_root_list_ ) {
		      free(cq_root_list_);
		      cq_root_list_ = (int64_t*)page_aligned_xmalloc(cq_distance_buf_length_ * sizeof(*cq_root_list_));
		   }
		}

		// using bitmap?
		if( next_bitmap_or_list_ ) {
	      const int64_t bitmap_width = get_bitmap_size_local();
	      assert(mpi.comm_r.size == mpi.size_2dc);
	      assert(work_buf_size_ >= mpi.size_2dc * bitmap_width * int64_t(sizeof(BitmapType)));
	      BitmapType* const restrict nq_bitmap = (BitmapType*)cache_aligned_xcalloc(bitmap_width * sizeof(*nq_bitmap));
	      BitmapType* recv_buffer_bitmap = (BitmapType*) work_buf_;
	      static_assert(sizeof(BitmapType) == sizeof(*cq_any_), "wrong data size");

	      // todo maybe don't parallelize?
#pragma omp parallel for schedule(static) if( nq_size > 1000 )
	      for( int64_t i = 0; i < nq_size; i++ ) {
	         const uint64_t v = nq_list_[i];
	         const uint64_t v_word = v >> LOG_NBPE;
	         const uint64_t v_bit = v & NBPE_MASK;
	         assert(v_word < uint64_t(bitmap_width));
#pragma omp atomic update
	         nq_bitmap[v_word] |= uint64_t(1) << v_bit;
	      }


#if ENABLE_MY_ALLGATHER == 1
         MpiCol::my_allgather(nq_bitmap, bitmap_width, recv_buffer_bitmap, mpi.comm_r);
#elif ENABLE_MY_ALLGATHER == 2
         my_allgather_2d(nq_bitmap, bitmap_width, recv_buffer_bitmap, mpi.comm_r);
#else
         MPI_Allgather(nq_bitmap, bitmap_width, get_mpi_type(nq_bitmap[0]), recv_buffer_bitmap, bitmap_width, get_mpi_type(nq_bitmap[0]), mpi.comm_2dr);
#endif
         free(nq_bitmap);
         cq_any_ = recv_buffer_bitmap;
         work_buf_state_ = Work_buf_state::cq;
		}
		else {
		   update_work_buf(int64_t(cq_size_) * int64_t(sizeof(TwodVertex)));
	      TwodVertex* recv_buf = (TwodVertex*) work_buf_;

#if ENABLE_MY_ALLGATHER == 1
         MpiCol::my_allgatherv(nq_list_, nq_size, recv_buf, recv_size, recv_off, mpi.comm_r);
#elif ENABLE_MY_ALLGATHER == 2
         my_allgatherv_2d(nq_list_, nq_size, recv_buf, recv_size, recv_off, mpi.comm_r);
#else
         MPI_Allgatherv(nq_list_, nq_size, MpiTypeOf<TwodVertex>::type, recv_buf, recv_size, recv_off, MpiTypeOf<TwodVertex>::type, mpi.comm_r.comm);
#endif
         cq_any_ = recv_buf;
         work_buf_state_ = Work_buf_state::cq;
		}

      /* NOTE: just for consistency */
      float* recv_buf_weight = cq_distance_list_;

#if ENABLE_MY_ALLGATHER == 1
		MpiCol::my_allgatherv(nq_distance_list_, nq_size, recv_buf_weight, recv_size, recv_off, mpi.comm_r);
#elif ENABLE_MY_ALLGATHER == 2
		my_allgatherv_2d(nq_distance_list_, nq_size, recv_buf_weight, recv_size, recv_off, mpi.comm_r);
#else
		MPI_Allgatherv(nq_distance_list_, nq_size, MpiTypeOf<float>::type, recv_buf_weight, recv_size, recv_off, MpiTypeOf<float>::type, mpi.comm_r.comm);
#endif
		VERBOSE(g_expand_list_comm += cq_size_ * sizeof(TwodVertex));
		assert(cq_distance_list_ == recv_buf_weight);

		if( nq_root_list_ ) {
		   assert(cq_root_list_);
#if ENABLE_MY_ALLGATHER == 1
		   MpiCol::my_allgatherv(nq_root_list_, nq_size, cq_root_list_, recv_size, recv_off, mpi.comm_r);
#elif ENABLE_MY_ALLGATHER == 2
      my_allgatherv_2d(nq_root_list_, nq_size, cq_root_list_, recv_size, recv_off, mpi.comm_r);
#else
         MPI_Allgatherv(nq_root_list_, nq_size, MpiTypeOf<int64_t>::type, cq_root_list_, recv_size, recv_off, MpiTypeOf<int64_t>::type, mpi.comm_r.comm);
#endif
		}
	}

	void top_down_expand() {
		TRACER(td_expand);
		// expand NQ within a processor column
		// convert NQ to a SRC format
		assert(!next_bitmap_or_list_);
		const TwodVertex shifted_c = TwodVertex(mpi.rank_2dc) << graph_.local_bits_;
		const int nq_size = top_down_make_nq(false, shifted_c);
		top_down_expand_nq(nq_size);
	}


	// expands current bucket vertices (and distances)
   void top_down_expand_bucket(int64_t& global_nq_size) {
#if VERBOSE_MODE
      const double expand_start_time = MPI_Wtime();
#endif

      TRACER(td_expand);
      const TwodVertex shifted_c = TwodVertex(mpi.rank_2dc) << graph_.local_bits_;
      const int nq_size = bucket_make_nq_list(false, shifted_c);

      int64_t send_nq_size = nq_size;
      int64_t nq_sum;
      MPI_Allreduce(&send_nq_size, &nq_sum, 1, MpiTypeOf<int64_t>::type, MPI_SUM, mpi.comm_2d);
      global_nq_size = nq_sum;

      if( global_nq_size > 0 ) {
         // expand NQ within processor column
         top_down_expand_nq(nq_size);

         if( mpi.isMaster() ) {
            if( is_bellman_ford_ )
               std::cout << "BFORD phase ";
            else if( is_light_phase_ )
               std::cout << "LIGHT phase ";
            else
               std::cout << "HEAVY phase ";

            std::cout << "next bucket (initial) size: " << global_nq_size;

            if( !is_light_phase_ )
               std::cout << "  (light iterations: " << current_phase_ << ") \n";
            else
               std::cout << '\n';
         }
      }

#if VERBOSE_MODE
      assert((MPI_Wtime() - expand_start_time) >= 0.0);
      profiling::expand_buckets_time += (MPI_Wtime() - expand_start_time);
#endif
   }


#ifdef USE_BOTTOM_UP
	void top_down_switch_expand(bool bitmap_or_list) {
		TRACER(td_sw_expand);
		// expand NQ within a processor row
		if(bitmap_or_list) {
			// bitmap
			expand_visited_bitmap();
		}
		else {
			throw "Not implemented";
		}
	}

	template <typename BitmapF>
	int make_list_from_bitmap(bool with_z, TwodVertex shifted_rc, BitmapF bmp, int bitmap_width, TwodVertex* outbuf) {
		int size_z = with_z ? mpi.size_z : 1;
		int rank_z = with_z ? mpi.rank_z : 0;

		const int max_threads = omp_get_max_threads();
		const int node_threads = max_threads * size_z;

		int th_offset_storage[max_threads+1];
		int *th_offset = with_z ? s_.offset : th_offset_storage;

		int result_size = 0;
#pragma omp parallel
		{
			int tid = omp_get_thread_num() + max_threads * rank_z;
			int count = 0;
#pragma omp for schedule(static) nowait
			for(int i = 0; i < bitmap_width; ++i) {
				count += __builtin_popcountl(bmp(i));
			}
			th_offset[tid+1] = count;
#pragma omp barrier
#pragma omp single
			{
				if(with_z) s_.sync->barrier();
				if(rank_z == 0) {
					th_offset[0] = 0;
					for(int i = 0; i < node_threads; ++i) {
						th_offset[i+1] += th_offset[i];
					}
					assert (th_offset[node_threads] <= int(bitmap_width*sizeof(BitmapType)/sizeof(TwodVertex)*size_z));
				}
				if(with_z) s_.sync->barrier();
				result_size = th_offset[node_threads];
			} // implicit barrier

			TwodVertex* dst = outbuf + th_offset[tid];
#pragma omp for schedule(static) nowait
			for(int i = 0; i < bitmap_width; ++i) {
				BitmapType bmp_i = bmp(i);
				while(bmp_i != 0) {
					TwodVertex bit_idx = __builtin_ctzl(bmp_i);
					*(dst++) = (i * NBPE + bit_idx) | shifted_rc;
					bmp_i &= bmp_i - 1;
				}
			}
			assert ((dst - outbuf) == th_offset[tid+1]);
		} // implicit barrier
		if(with_z) s_.sync->barrier();

		return result_size;
	}

	struct NQBitmapCombiner {
		BitmapType* new_visited;
		BitmapType* old_visited;
		NQBitmapCombiner(ThisType* this__)
			: new_visited((BitmapType*)this__->new_visited_)
			, old_visited((BitmapType*)this__->old_visited_) { }
		BitmapType operator ()(int i) { return new_visited[i] & ~(old_visited[i]); }
	};

	int bottom_up_make_nq_list(bool with_z, TwodVertex shifted_rc, TwodVertex* outbuf) {
		TRACER(bu_make_nq_list);
		const int bitmap_width = get_bitmap_size_local();
		int node_nq_size;

		if(bitmap_or_list_) {
			NQBitmapCombiner NQBmp(this);
			node_nq_size = make_list_from_bitmap(with_z, shifted_rc, NQBmp, bitmap_width, outbuf);
		}
		else {
			int size_z = with_z ? mpi.size_z : 1;
			int rank_z = with_z ? mpi.rank_z : 0;
			TwodVertex* new_vis_p[BU_SUBSTEP];
			get_visited_pointers(new_vis_p, BU_SUBSTEP, new_visited_, BU_SUBSTEP);
			TwodVertex* old_vis_p[BU_SUBSTEP];
			get_visited_pointers(old_vis_p, BU_SUBSTEP, old_visited_, BU_SUBSTEP);
			const int num_parts = BU_SUBSTEP * size_z;
			int part_offset_storage[num_parts+1];
			int *part_offset = with_z ? s_.offset : part_offset_storage;

			for(int i = 0; i < BU_SUBSTEP; ++i) {
				part_offset[rank_z*BU_SUBSTEP+1+i] = old_visited_list_size_[i] - new_visited_list_size_[i];
			}
			if(with_z) s_.sync->barrier();
			if(rank_z == 0) {
				part_offset[0] = 0;
				for(int i = 0; i < num_parts; ++i) {
					part_offset[i+1] += part_offset[i];
				}
				assert (part_offset[num_parts] <= int(bitmap_width*sizeof(BitmapType)/sizeof(TwodVertex)*2));
			}
			if(with_z) s_.sync->barrier();
			node_nq_size = part_offset[num_parts];

			bool mt = int(node_nq_size*sizeof(TwodVertex)) > 16*1024*mpi.size_z;
#ifndef NDEBUG
			int dbg_inc0 = 0, dbg_exc0 = 0, dbg_inc1 = 0, dbg_exc1 = 0;
#pragma omp parallel if(mt) reduction(+:dbg_inc0, dbg_exc0, dbg_inc1, dbg_exc1)
#else
#pragma omp parallel if(mt)
#endif
			for(int i = 0; i < BU_SUBSTEP; ++i) {
				int max_threads = omp_get_num_threads(); // Place here because this region may be executed sequential.
				int tid = omp_get_thread_num();
				TwodVertex substep_base = graph_.num_local_verts_ / BU_SUBSTEP * i;
				TwodVertex* dst = outbuf + part_offset[rank_z*BU_SUBSTEP+i];
				TwodVertex *new_vis = new_vis_p[i], *old_vis = old_vis_p[i];
				int start, end, old_size = old_visited_list_size_[i], new_size = new_visited_list_size_[i];
				get_partition(old_size, max_threads, tid, start, end);
				int new_vis_start = std::lower_bound(
						new_vis, new_vis + new_size, old_vis[start]) - new_vis;
				int new_vis_off = new_vis_start;
				int dst_off = start - new_vis_start;

				for(int c = start; c < end; ++c) {
					if(new_vis[new_vis_off] == old_vis[c]) {
						++new_vis_off;
					}
					else {
						dst[dst_off++] = (old_vis[c] + substep_base) | shifted_rc;
					}
				}

#ifndef NDEBUG
				if(i == 0) {
					dbg_inc0 += dst_off - (start - new_vis_start);
					dbg_exc0 += new_vis_off - new_vis_start;
				}
				else if(i == 1) {
					dbg_inc1 += dst_off - (start - new_vis_start);
					dbg_exc1 += new_vis_off - new_vis_start;
				}
#endif
			}
#ifndef NDEBUG
			assert(dbg_inc0 == old_visited_list_size_[0] - new_visited_list_size_[0]);
			assert(dbg_exc0 == new_visited_list_size_[0]);
			assert(dbg_inc1 == old_visited_list_size_[1] - new_visited_list_size_[1]);
			assert(dbg_exc1 == new_visited_list_size_[1]);
#endif
			if(with_z) s_.sync->barrier();
		}

		return node_nq_size;
	}

	void bottom_up_expand_nq_list() {
		TRACER(bu_expand_nq_list);
		assert (mpi.isYdimAvailable() || (visited_buffer_orig_ == visited_buffer_));
		const int lgl = graph_.local_bits_;
		const int L = graph_.num_local_verts_;

		const int node_nq_size = bottom_up_make_nq_list(
				mpi.isYdimAvailable(), TwodVertex(mpi.rank_2dr) << lgl, (TwodVertex*)visited_buffer_orig_);

		const int recv_nq_size = expand_visited_list(node_nq_size);

#pragma omp parallel if(recv_nq_size > 1024*16)
		{
			const int max_threads = omp_get_num_threads(); // Place here because this region may be executed sequential.
			const int node_threads = max_threads * mpi.size_z;
			const int tid = omp_get_thread_num() + max_threads * mpi.rank_z;
			int64_t begin, end;
			get_partition(recv_nq_size, nq_recv_buf_, get_msb_index(NBPE), node_threads, tid, begin, end);

			for(int i = begin; i < end; ++i) {
				SeparatedId dst(nq_recv_buf_[i]);
				TwodVertex compact = dst.compact(lgl, L);
				TwodVertex word_idx = compact >> LOG_NBPE;
				int bit_idx = compact & NBPE_MASK;
				shared_visited_[word_idx] |= BitmapType(1) << bit_idx;
			}
		} // implicit barrier
		if(mpi.isYdimAvailable()) s_.sync->barrier();
	}

	void bottom_up_expand(bool bitmap_or_list) {
		TRACER(bu_expand);
		if(bitmap_or_list) {
			// bitmap
			assert (bitmap_or_list_);
			expand_visited_bitmap();
		}
		else {
			// list
			bottom_up_expand_nq_list();
		}
	}

	void bottom_up_switch_expand(bool bitmap_or_list) {
		TRACER(bu_sw_expand);
		if(bitmap_or_list) {
			// new_visited - old_visited = nq
			const int bitmap_width = get_bitmap_size_local();
			BitmapType* new_visited = (BitmapType*)new_visited_;
			BitmapType* old_visited = (BitmapType*)old_visited_;
#pragma omp parallel for
			for(int i = 0; i < bitmap_width; ++i) {
				new_visited[i] &= ~(old_visited[i]);
			}
			// expand nq
			expand_nq_bitmap();
		}
		else {
			// visited_buffer_ is used as a temporal buffer
			TwodVertex* nq_list = (TwodVertex*)visited_buffer_;
			int nq_size = bottom_up_make_nq_list(
					false, TwodVertex(mpi.rank_2dc) << graph_.local_bits_, (TwodVertex*)nq_list);

			assert(0 && "update nq_distance_list");
			top_down_expand_nq(nq_size);
		}
	}
#endif

	//-------------------------------------------------------------//
	// top-down search
	//-------------------------------------------------------------//


	// is the target already settled?
	// todo
	// TODO: there is a duplicate of this method in sssp_state, merge them!
   bool top_down_target_is_settled(int64_t tgt, int r_bits, int lgl, int64_t L) const {
      const TwodVertex bit_idx = SeparatedId(SeparatedId(tgt).low(r_bits + lgl)).compact(lgl, L);
      return ( vertices_isSettled_[bit_idx >> PRM::LOG_NBPE] & (BitmapType(1) << (bit_idx & PRM::NBPE_MASK)) );
   }

	void top_down_send(int64_t tgt, float tgt_weight, int lgl, int r_mask,
			LocalPacket* packet_array, int64_t src, int64_t root
#if PROFILING_MODE
			, profiling::TimeSpan& ts_commit
#endif
	) {
	   const int dest = (tgt >> lgl) & r_mask;
		LocalPacket& pk = packet_array[dest];

		// is the packet full?
		if(pk.length > LocalPacket::TOP_DOWN_LENGTH-4) { // low probability
			PROF(profiling::TimeKeeper tk_commit);
			td_comm_.put(pk.data.t, pk.length, dest);
			PROF(ts_commit += tk_commit);
			pk.src = -1;
			pk.length = 0;
		}
		if(pk.src != src) { // TODO: use conditional branch
			pk.src = src;
			//std::cout << "rank" << mpi.rank_2d << " new source: " << src << '\n';

			if( !is_presolve_mode_ ) {
	         assert(!((src >> 32) & 0x80000000u));
			   pk.data.t[pk.length++] = (src >> 32) | 0x80000000u;
			   pk.data.t[pk.length++] = (uint32_t)src;
			} else {
            assert(!((root >> 32) & 0x80000000u));
            pk.data.t[pk.length++] = (root >> 32) | 0x80000000u;
            pk.data.t[pk.length++] = (uint32_t)root;
			}
		}
		pk.data.t[pk.length++] = tgt & ((uint32_t(1) << lgl) - 1);
		pk.data.t[pk.length++] = castFloatToUInt32(tgt_weight);
		//printf("rank%d sends %u,%f (length=%d) to row%d \n", mpi.rank_2d, uint32_t(tgt & ((uint32_t(1) << lgl) - 1)), tgt_weight, pk.length, dest);
	}

	void top_down_send_large(const int64_t* restrict edge_array, int64_t start, int64_t end,
			int lgl, int r_mask, int64_t src, int64_t root, float dist, bool is_heavy)
	{
		assert (end >= start);
		assert(!(src & int64_t(1) << 63));
		if( end == start )
		   return;

		int64_t header;
		if( is_presolve_mode_  )
		   header = is_heavy ? (root | int64_t(1) << 63) : root;
		else
		   header = is_heavy ? (src | int64_t(1) << 63) : src;

		for(int i = 0; i < mpi.size_2dr; ++i) {
			if(start >= end) break;
			const int s_dest = (edge_array[start] >> lgl) & r_mask;
			if(s_dest > i) continue;

			// search the destination change point with binary search
			uint64_t left = start;
			uint64_t right = end;
			uint64_t next = std::min<uint64_t>(left + (right - left) / (mpi.size_2dr - i) * 2, end - 1);
			do {
				const int dest = (edge_array[next] >> lgl) & r_mask;
				if(dest > i) {
					right = next;
				}
				else {
					left = next;
				}
				next = (left + right) / 2;
			} while(left < next);
			// start ... right -> i
			td_comm_.put_ptr(start, right - start, header, dist, i);
			start = right;
		}
		assert(start == end);
	}


	void top_down_parallel_section() {
		TRACER(td_par_sec);
		PROF(profiling::TimeKeeper tk_all);
		const bool clear_packet_buffer = packet_buffer_is_dirty_;
		packet_buffer_is_dirty_ = false;
		TwodVertex* restrict cq_rowsums = nullptr;

#if TOP_DOWN_SEND_LB == 2
#define IF_LARGE_EDGE if(e_end_phase - e_start > PRM::TOP_DOWN_PENDING_WIDTH/10)
#define ELSE else
#else
#define IF_LARGE_EDGE
#define ELSE
#endif

		if( bitmap_or_list_ ) {
         const uint64_t bitmap_size = get_bitmap_size_local() * mpi.size_2dc;
         const BitmapType* const restrict cq_bitmap = (BitmapType*)cq_any_;
		   cq_rowsums = (TwodVertex*)cache_aligned_xmalloc(bitmap_size * sizeof(*cq_rowsums));

#pragma omp parallel for schedule(static)
		   for( uint64_t i = 1; i < bitmap_size; i++ ) {
		      cq_rowsums[i] = __builtin_popcountl(cq_bitmap[i - 1]);
		   }

		   cq_rowsums[0] = 0;
		   for( uint64_t i = 2; i < bitmap_size; i++ )
		      cq_rowsums[i] += cq_rowsums[i - 1];
		}

		debug("begin parallel");
#pragma omp parallel
		{
			SET_OMP_AFFINITY;
			PROF(profiling::TimeKeeper tk_all);
			PROF(profiling::TimeSpan ts_commit);
			VERBOSE(int64_t num_edge_relax = 0);
			VERBOSE(int64_t num_large_edge = 0);
			const int64_t* const restrict edge_array = graph_.edge_array_;
#if TOP_DOWN_SEND_LB != 1
			const float* const restrict edge_weight_array = graph_.edge_weight_array_;
         const int r_bits = graph_.r_bits_;
         const bool with_settled = has_settled_vertices_;
#endif
			LocalPacket* const packet_array = thread_local_buffer_[omp_get_thread_num()]->fold_packet;
			if(clear_packet_buffer) {
				for(int target = 0; target < mpi.size_2dr; ++target) {
					packet_array[target].src = -1;
					packet_array[target].length = 0;
				}
			}
			const int lgl = graph_.local_bits_;
			const int r_mask = (1 << graph_.r_bits_) - 1;
			const int P = mpi.size_2d;
			const int R = mpi.size_2dr;
			const int r = mpi.rank_2dr;
			const uint32_t local_mask = (uint32_t(1) << lgl) - 1;
			const int64_t L = graph_.num_local_verts_;

			if( bitmap_or_list_ ) {
			   assert(is_bellman_ford_ && "todo maybe support other modes?");
				const BitmapType* const restrict cq_bitmap = (BitmapType*)cq_any_;
				const int64_t bitmap_size_local = get_bitmap_size_local();
				const int64_t bitmap_size = bitmap_size_local * mpi.size_2dc;
	#pragma omp for
				for(int64_t word_idx = 0; word_idx < bitmap_size; ++word_idx) {
					const BitmapType cq_bit_i = cq_bitmap[word_idx];
					if(cq_bit_i == BitmapType(0)) continue;

					const BitmapType row_bitmap_i = graph_.row_bitmap_[word_idx];
					const TwodVertex bmp_row_sum = graph_.row_sums_[word_idx];
               const TwodVertex cq_rowsum = cq_rowsums[word_idx];

               BitmapType bit_flags = cq_bit_i & row_bitmap_i;
					while(bit_flags != BitmapType(0)) {
						const BitmapType cq_bit = bit_flags & (-bit_flags);
						const BitmapType low_mask = cq_bit - 1;
						bit_flags &= ~cq_bit;
						const TwodVertex src_c = word_idx / bitmap_size_local; // TODO:
						const TwodVertex non_zero_off = bmp_row_sum + __builtin_popcountl(row_bitmap_i & low_mask);
						const int64_t src_orig = int64_t(graph_.orig_vertexes_[non_zero_off]) * P + src_c * R + r;
						const int64_t e_start = graph_.row_starts_[non_zero_off];
						const int64_t e_end = graph_.row_starts_[non_zero_off+1];
#if TOP_DOWN_SEND_LB == 2
						const int64_t e_end_phase = e_end;
#endif

                  const TwodVertex cq_off = cq_rowsum + __builtin_popcountl(cq_bit_i & low_mask);
                  const float distance = cq_distance_list_[cq_off];
                  const int64_t root = is_presolve_mode_ ? cq_root_list_[cq_off] : (-1);

                  IF_LARGE_EDGE
#if TOP_DOWN_SEND_LB > 0
                  {
                     const int64_t e_start_heavy = graph_.row_starts_heavy_[non_zero_off];
                     top_down_send_large(edge_array, e_start, e_start_heavy, lgl, r_mask, src_orig, root, distance, false);
                     top_down_send_large(edge_array, e_start_heavy, e_end, lgl, r_mask, src_orig, root, distance, true);
                     VERBOSE(num_large_edge += e_end - e_start);
                  }
#endif // #if TOP_DOWN_SEND_LB > 0
                  ELSE
#if TOP_DOWN_SEND_LB != 1
                  {
                     for( int64_t e = e_start; e < e_end; ++e ) {
                        const int64_t tgt = edge_array[e];
                        if( top_down_target_is_settled(tgt, r_bits, lgl, L) )
                           continue;

                        top_down_send(tgt, edge_weight_array[e] + distance, lgl, r_mask, packet_array, src_orig, root
                           profiling_commit(ts_commit) );
                     }
                  }
#endif // #if TOP_DOWN_SEND_LB != 1
						VERBOSE(num_edge_relax += e_end - e_start + 1);
					} // while(bit_flags != BitmapType(0)) {
				} // #pragma omp for // implicit barrier
			}
			else
			{
            const TwodVertex* const restrict cq_list = cq_any_;
            const float* const restrict cq_distance_list = cq_distance_list_;
#if TOP_DOWN_SEND_LB != 0
            const bool is_light_phase_proper = is_light_phase_ && !is_bellman_ford_;
#endif
            const float bucket_upper = (delta_epoch_ + 1.0) * delta_step_;
            const bool is_bellman_ford = is_bellman_ford_;
            const bool is_light_phase = is_light_phase_;

#pragma omp for
				for(int64_t i = 0; i < int64_t(cq_size_); ++i) {
					const SeparatedId src(cq_list[i]);
					const TwodVertex src_c = src.value >> lgl;
					const TwodVertex compact = src_c * L + (src.value & local_mask);
					const TwodVertex word_idx = compact >> LOG_NBPE;
					const int bit_idx = compact & NBPE_MASK;
					const BitmapType row_bitmap_i = graph_.row_bitmap_[word_idx];
					const BitmapType mask = BitmapType(1) << bit_idx;

					if(row_bitmap_i & mask) {
						const BitmapType low_mask = (BitmapType(1) << bit_idx) - 1;
						const TwodVertex non_zero_off = graph_.row_sums_[word_idx] + __builtin_popcountl(graph_.row_bitmap_[word_idx] & low_mask);
						const int64_t src_orig = int64_t(graph_.orig_vertexes_[non_zero_off]) * P + src_c * R + r;
						const int64_t root = is_presolve_mode_ ? cq_root_list_[i] : (-1);
					   const int64_t e_start = graph_.row_starts_[non_zero_off];
                  const int64_t e_end = graph_.row_starts_[non_zero_off + 1];
                  const float distance = cq_distance_list[i];
#if TOP_DOWN_SEND_LB == 2
                  const int64_t e_end_phase = is_light_phase_proper ? graph_.row_starts_heavy_[non_zero_off] : e_end;
#endif

                  IF_LARGE_EDGE
#if TOP_DOWN_SEND_LB > 0
                  {
                     const int64_t e_start_heavy = graph_.row_starts_heavy_[non_zero_off];
                     if( is_light_phase_proper ) {
                        top_down_send_large(edge_array, e_start, e_start_heavy, lgl, r_mask, src_orig, root, distance, false);
                     }
                     else {
                        top_down_send_large(edge_array, e_start, e_start_heavy, lgl, r_mask, src_orig, root, distance, false);
                        top_down_send_large(edge_array, e_start_heavy, e_end, lgl, r_mask, src_orig, root, distance, true);
                     }
                     VERBOSE(num_large_edge += e_end - e_start);
                  }
#endif // #if TOP_DOWN_SEND_LB > 0
                  ELSE
#if TOP_DOWN_SEND_LB != 1
                  {
                     if( is_bellman_ford ) {
                        assert(with_settled);
                        for( int64_t e = e_start; e < e_end; ++e ) {
                           const int64_t tgt = edge_array[e];
                           if( top_down_target_is_settled(tgt, r_bits, lgl, L) )
                              continue;

                           top_down_send(tgt, edge_weight_array[e] + distance, lgl, r_mask, packet_array,
                                 src_orig, root profiling_commit(ts_commit));
                        }
                     }
                     else if( is_light_phase ) {
                        const int64_t e_start_heavy = graph_.row_starts_heavy_[non_zero_off];
                        for( int64_t e = e_start; e < e_start_heavy; ++e ) {
                           const float dist_new = edge_weight_array[e] + distance;
                           if( dist_new >= bucket_upper )
                              continue;

                           const int64_t tgt = edge_array[e];
                           if( with_settled && top_down_target_is_settled(tgt, r_bits, lgl, L) )
                              continue;

                           top_down_send(tgt, dist_new, lgl, r_mask, packet_array,
                                 src_orig, root profiling_commit(ts_commit));
                        }
                     }
                     else { // heavy phase
                        const int64_t e_start_heavy = graph_.row_starts_heavy_[non_zero_off];
                        for( int64_t e = e_start; e < e_start_heavy; ++e ) {
                           const float dist_new = edge_weight_array[e] + distance;
                           if( comp::isLT(dist_new, bucket_upper) )
                              continue;

                           const int64_t tgt = edge_array[e];
                           if( with_settled && top_down_target_is_settled(tgt, r_bits, lgl, L) )
                              continue;

                           top_down_send(tgt, dist_new, lgl, r_mask, packet_array,
                                 src_orig, root profiling_commit(ts_commit));
                        }

                        for( int64_t e = e_start_heavy; e < e_end; ++e ) {
                           const float dist_new = edge_weight_array[e] + distance;
                           assert(!comp::isLT(dist_new, bucket_upper));

                           const int64_t tgt = edge_array[e];
                           if( with_settled && top_down_target_is_settled(tgt, r_bits, lgl, L) )
                              continue;

                           top_down_send(tgt, dist_new, lgl, r_mask, packet_array,
                                 src_orig, root profiling_commit(ts_commit));
                        }
                     }
                     VERBOSE(num_edge_relax += e_end - e_start + 1);
                  }
#endif // #if TOP_DOWN_SEND_LB != 1
					} // if(row_bitmap_i & mask) {
				} // #pragma omp for // implicit barrier
			}

			// flush buffer
#pragma omp for
			for(int target = 0; target < mpi.size_2dr; ++target) {
			   const int num_threads = omp_get_num_threads();
				for(int i = 0; i < num_threads; ++i) {
					LocalPacket* packet_array = thread_local_buffer_[i]->fold_packet;
					LocalPacket& pk = packet_array[target];
					if(pk.length > 0) {
						PROF(profiling::TimeKeeper tk_commit);
						td_comm_.put(pk.data.t, pk.length, target);
						PROF(ts_commit += tk_commit);
						pk.src = -1;
						pk.length = 0;
					}
				}
			} // #pragma omp for
			PROF(profiling::TimeSpan ts_all; ts_all += tk_all; ts_all -= ts_commit);
			PROF(extract_edge_time_ += ts_all);
			PROF(commit_time_ += ts_commit);
			VERBOSE(__sync_fetch_and_add(&num_edge_top_down_, num_edge_relax));
			VERBOSE(__sync_fetch_and_add(&num_td_large_edge_, num_large_edge));
		} // #pragma omp parallel reduction(+:num_edge_relax)
#undef IF_LARGE_EDGE
#undef ELSE

		if( cq_rowsums ) free(cq_rowsums);
		PROF(parallel_reg_time_ += tk_all);
		debug("finished parallel");
	}


	void top_down_search() {
		TRACER(td);

		td_comm_.prepare();
		top_down_parallel_section();

#if TOP_DOWN_SEND_LB == 0
		td_comm_.run_buffer(graph_, state, vertices_pos_);
#elif TOP_DOWN_SEND_LB == 1
		SsspState state = get_state();
		td_comm_.run_ptr(graph_, state, vertices_pos_);
#else
      SsspState state = get_state();
      td_comm_.run_with_both(graph_, state, vertices_pos_);
#endif

		PROF(profiling::TimeKeeper tk_all);
		// flush NQ buffer and count NQ total
		const int max_threads = omp_get_max_threads();
		nq_size_ = nq_.stack_.size() * QueuedVertexes::SIZE;

		for(int tid = 0; tid < max_threads; ++tid) {
			ThreadLocalBuffer* tlb = thread_local_buffer_[tid];
			QueuedVertexes* buf = tlb->cur_buffer;
			if(buf != NULL) {
				nq_size_ += buf->length;
				nq_.push(buf);
			}
			tlb->cur_buffer = NULL;
		}

		// todo make it less hacky
		if( current_level_ == -1 ) {
		   assert(dist_presol_ && pred_presol_);
		   assert(is_presolve_mode_);
		   if( mpi.isMaster() )
		      std::cout << "starting new presolving round" << '\n';
#if 0
	      const uint64_t num_local_verts = uint64_t(graph_.num_local_verts_);

	      for( uint64_t i = 0; i < num_local_verts; i++ ) {
	         if( dist_[i] < comp::infinity ) {
               dist_presol_[i] = dist_[i];
               pred_presol_[i] = pred_[i];
               //printf("pred=%d \n", pred_[i]);
	         }
	      }
#endif
		}

      PROF(seq_proc_time_ += tk_all);
      PROF(MPI_Barrier(mpi.comm_2d));
      PROF(fold_competion_wait_ += tk_all);

		if( !is_light_phase_ ) {
		   assert(!is_bellman_ford_);
		   assert(0 == nq_size_);
		   global_nq_size_ = 0;
		}
		else {
         int64_t send_nq_size = nq_size_;
         MPI_Allreduce(&send_nq_size, &global_nq_size_, 1, MpiTypeOf<int64_t>::type, MPI_SUM, mpi.comm_2d);
		}

      PROF(gather_nq_time_ += tk_all);
	}


   void top_down_receive_ptr_presolve(uint32_t* stream, int length, int thread_id) {
      assert(thread_id >= 0);
      assert(pred_presol_ && dist_presol_);
      // todo current_level_ >= 0 is a bad hack to avoid updates during the first push
      const bool is_first_round = (current_level_ == -1);

      for( int i = 0; i < length; i++ ) {
         const int64_t pred_v = (int64_t(stream[i] & 0xFFFF) << 32) | stream[i+1];
         assert(pred_v >= 0);
         assert(!(stream[i + 2] & 0x80000000u));

         const int length_i = stream[i + 2];
         const int row_start = i + 3;
         const int row_end = row_start + length_i;
         assert(length_i % 2 == 0);
         assert(length_i > 0);

          for( int c = row_start; c < row_end; c += 2 ) {
             const LocalVertex tgt_local = stream[c];
             assert(tgt_local == (stream[c] & ((LocalVertex(1) << graph_.local_bits_) - 1)));
             const float weight = castUInt32ToFloat(stream[c + 1]);
             assert(0 <= tgt_local && tgt_local < graph_.num_local_verts_);

             if( is_first_round ) {
 #if USE_DISTANCE_LOCKS
                omp_set_lock(&vertices_locks_[tgt_local]);
 #endif
                if( pred_presol_[tgt_local] == -1 || weight > dist_presol_[tgt_local] ) {
                   dist_presol_[tgt_local] = weight;
                   pred_presol_[tgt_local] = pred_v;
                }
 #if USE_DISTANCE_LOCKS
                omp_unset_lock(&vertices_locks_[tgt_local]);
 #endif
                continue;
             }


             if( pred_v == pred_presol_[tgt_local] && weight < dist_presol_[tgt_local] ) {
 #if USE_DISTANCE_LOCKS
                omp_set_lock(&vertices_locks_[tgt_local]);
 #endif
                if( comp::isLT(weight, dist_presol_[tgt_local]) ) {
                 //  std::cout << " PTR marking vertex dist=" << dist_presol_[tgt_local] << " dist_pred=" << pred_presol_[tgt_local] << '\n';
                   dist_presol_[tgt_local] = -1.0;
                }

 #if USE_DISTANCE_LOCKS
                omp_unset_lock(&vertices_locks_[tgt_local]);
 #endif
             }
          }
          i += 2 + length_i;
      }
   }

	template <bool with_nq>
	void top_down_row_receive(TopDownRow* rows, int num_rows) {
		const int num_threads = omp_get_max_threads();
		const int num_splits = num_threads * 8;
		// process from tail because computation cost is higher in tail
		volatile int procces_counter = num_splits - 1;
		//volatile int procces_counter = 0;

#pragma omp parallel
		{
			TRACER(td_recv);
			PROF(profiling::TimeKeeper tk_all);
			const int tid = omp_get_thread_num();
			ThreadLocalBuffer* tlb = thread_local_buffer_[tid];
			QueuedVertexes* buf = tlb->cur_buffer;
			if(buf == NULL) buf = nq_empty_buffer_.get();
			int64_t* restrict const pred = pred_;
	      float* restrict const dist = dist_;

			while(true) {
				const int split = __sync_fetch_and_add(&procces_counter, -1);
				if(split < 0) break;

				for(int r = 0; r < num_rows; ++r) {
					const uint32_t* const ptr = rows[r].ptr;
					const int length = rows[r].length;
					const int64_t pred_v = rows[r].src;
					assert(length % 2 == 0);

				   int width_per_split = (length + num_splits - 1) / num_splits;
					assert((width_per_split & 1) == (width_per_split %  2));
					if( width_per_split & 1 )
					   width_per_split++;
				   const int off_start = std::min(length, width_per_split * split);
					const int off_end = std::min(length, off_start + width_per_split);
					assert(off_start % 2 == 0 && off_end % 2 == 0);

					for( int i = off_start; i < off_end;  i+= 2 ) {
						const LocalVertex tgt_local = ptr[i];

						// todo extra inline method! currently duplicate!
		             const float weight = castUInt32ToFloat(ptr[i + 1]);
		             assert(0 <= tgt_local && tgt_local < graph_.num_local_verts_);

		             if( weight < dist[tgt_local] ) {
		 #if USE_DISTANCE_LOCKS
		                if( !with_nq )
		                   omp_set_lock(&vertices_locks_[tgt_local]);
		 #endif

		                // todo better have a relative comparison here?
		                if( comp::isLT(weight, dist[tgt_local]) ) {
		                   if( !with_nq ) {
		                      dist[tgt_local] = weight;
		                      pred[tgt_local] = pred_v;
		 #if USE_DISTANCE_LOCKS
		                      omp_unset_lock(&vertices_locks_[tgt_local]);
		 #endif
		                   }
		                   assert(comp::isLE(delta_epoch_ * delta_step_, weight)); // weight should not be in lower bucket

		                   if( with_nq ) {
		                      if(buf->full()) {
		                         nq_.push(buf); buf = nq_empty_buffer_.get();
		                      }
		                      buf->append_nocheck(tgt_local, pred_v, weight);
		                   }
		                }
		 #if USE_DISTANCE_LOCKS
		                else if( !with_nq )  {
		                   omp_unset_lock(&vertices_locks_[tgt_local]);
		                }
		 #endif
		             }
					}
				}
			}
			tlb->cur_buffer = buf;
			PROF(recv_proc_thread_large_time_ += tk_all);
		} // #pragma omp parallel
	}

   template <bool with_nq>
   void top_down_receive_ptr(uint32_t* stream, int length, TopDownRow* rows, volatile int* num_rows, int thread_id) {
      TRACER(td_recv);
      PROF(profiling::TimeKeeper tk_all);
      assert(thread_id >= 0);

      if( is_presolve_mode_ ) {
         top_down_receive_ptr_presolve(stream, length, thread_id);
      }

      ThreadLocalBuffer* const tlb = thread_local_buffer_[thread_id];
      QueuedVertexes* buf = tlb->cur_buffer;
      if(buf == NULL) buf = nq_empty_buffer_.get();
      int64_t* restrict const pred = pred_;
      float* restrict const dist = dist_;

      // ------------------- //
      for( int i = 0; i < length; i++ ) {
         const int64_t pred_v = (int64_t(stream[i] & 0xFFFF) << 32) | stream[i+1];
         assert(pred_v >= 0);
         assert(!(stream[i + 2] & 0x80000000u));

         const int length_i = stream[i + 2];
         const int row_start = i + 3;
         const int row_end = row_start + length_i;
         assert(length_i % 2 == 0);
         assert(length_i > 0);

#if TOP_DOWN_RECV_LB
          if(length_i < 2 * PRM::TOP_DOWN_PENDING_WIDTH) {
#endif // #if TOP_DOWN_RECV_LB
          for( int c = row_start; c < row_end; c += 2 ) {
             const LocalVertex tgt_local = stream[c];
             assert(tgt_local == (stream[c] & ((LocalVertex(1) << graph_.local_bits_) - 1)));
             const float weight = castUInt32ToFloat(stream[c + 1]);
             assert(0 <= tgt_local && tgt_local < graph_.num_local_verts_);
            //  printf("rank%d receive: %d,%f pred=%" PRId64 " \n", mpi.rank_2d, tgt_local, castUInt32ToFloat(stream[c + 1]), pred_v);
             //printf("pred_v=%u target=%u weight=%f i=%d\n", unsigned(pred_v), tgt_local, weight, i);

             if( weight < dist[tgt_local] ) {
 #if USE_DISTANCE_LOCKS
                if( !with_nq )
                   omp_set_lock(&vertices_locks_[tgt_local]);
 #endif

                // todo better have a relative comparison here?
                if( comp::isLT(weight, dist[tgt_local]) ) {
                   //printf("...updated node %d (%f->%f, src=%d) \n", tgt_orig, std::min(100.0f, dist[tgt_orig]), weight, unsigned(pred_v));

                   if( !with_nq ) {
                      dist[tgt_local] = weight;
                      pred[tgt_local] = pred_v;
 #if USE_DISTANCE_LOCKS
                      omp_unset_lock(&vertices_locks_[tgt_local]);
 #endif
                   }
                   assert(comp::isLE(delta_epoch_ * delta_step_, weight)); // weight should not be in lower bucket

                   if( with_nq ) {
                      if(buf->full()) {
                         nq_.push(buf); buf = nq_empty_buffer_.get();
                      }
                      buf->append_nocheck(tgt_local, pred_v, weight);
                   }
                }
 #if USE_DISTANCE_LOCKS
                else if( !with_nq )  {
                   omp_unset_lock(&vertices_locks_[tgt_local]);
                }
 #endif
             }
          }
#if TOP_DOWN_RECV_LB
          }
          else {
             int put_off = __sync_fetch_and_add(num_rows, 1);
             rows[put_off].length = length_i;
             rows[put_off].ptr = &stream[i+3];
             rows[put_off].src = pred_v;
          }
#endif // #if TOP_DOWN_RECV_LB
          i += 2 + length_i;
      }
      tlb->cur_buffer = buf;
      PROF(recv_proc_thread_time_ += tk_all);
   }


   void top_down_receive_presolve(uint32_t* stream, int length, int thread_id) {
      assert(thread_id >= 0);
      assert(pred_presol_ && dist_presol_);
      int64_t pred_v = -1;
      const bool is_first_round = (current_level_ == -1);
      // todo current_level_ >= 0 is a bad hack to avoid updates during the first push

      for( int i = 0; i < length; i+= 2 ) {
         const uint32_t v = stream[i];
         if(v & 0x80000000u) {
            const int64_t src = (int64_t(v & 0xFFFF) << 32) | stream[i+1];
            pred_v = src; // | (int64_t(cur_level) << 48);
            assert(!(v & 0x40000000u)); // currently not supported
         }
         else {
            assert (pred_v != -1);

            const LocalVertex tgt_local = v;
            assert(tgt_local == (v & ((LocalVertex(1) << graph_.local_bits_) - 1)));
            const float weight = castUInt32ToFloat(stream[i + 1]);
            assert(0 <= tgt_local && tgt_local < graph_.num_local_verts_);

            if( is_first_round ) {
#if USE_DISTANCE_LOCKS
               omp_set_lock(&vertices_locks_[tgt_local]);
#endif
               if( pred_presol_[tgt_local] == -1 || weight > dist_presol_[tgt_local] ) {
                  dist_presol_[tgt_local] = weight;
                  pred_presol_[tgt_local] = pred_v;
               }
#if USE_DISTANCE_LOCKS
               omp_unset_lock(&vertices_locks_[tgt_local]);
#endif
               continue;
            }

            if( pred_v == pred_presol_[tgt_local] && weight < dist_presol_[tgt_local] ) {
#if USE_DISTANCE_LOCKS
               omp_set_lock(&vertices_locks_[tgt_local]);
#endif
               if( comp::isLT(weight, dist_presol_[tgt_local]) ) {
                 // std::cout << " marking vertex dist=" << dist_presol_[tgt_local] << " dist_pred=" << pred_presol_[tgt_local] << '\n';
                  dist_presol_[tgt_local] = -1.0;
               }
#if USE_DISTANCE_LOCKS
               omp_unset_lock(&vertices_locks_[tgt_local]);
#endif
            }
         }
      }
   }



	template <bool with_nq>
	void top_down_receive(uint32_t* stream, int length, int thread_id) {
		TRACER(td_recv);
		PROF(profiling::TimeKeeper tk_all);
		assert(thread_id >= 0);

		if( is_presolve_mode_ ) {
		   top_down_receive_presolve(stream, length, thread_id);
		}

		ThreadLocalBuffer* const tlb = thread_local_buffer_[thread_id];
		QueuedVertexes* buf = tlb->cur_buffer;
		if(buf == NULL) buf = nq_empty_buffer_.get();
		//BitmapType* visited = (BitmapType*)new_visited_;
		int64_t* restrict const pred = pred_;
		float* restrict const dist = dist_;
	//	const int cur_level = current_level_;
		int64_t pred_v = -1;

		// ------------------- //
		for( int i = 0; i < length; i+= 2 ) {
			const uint32_t v = stream[i];
			if(v & 0x80000000u) {
				const int64_t src = (int64_t(v & 0xFFFF) << 32) | stream[i+1];
				pred_v = src; // | (int64_t(cur_level) << 48);
				assert(!(v & 0x40000000u)); // currently not supported
			}
			else {
				assert (pred_v != -1);

				const LocalVertex tgt_local = v;
				assert(tgt_local == (v & ((LocalVertex(1) << graph_.local_bits_) - 1)));
            const float weight = castUInt32ToFloat(stream[i + 1]);

            //printf("rank%d receive: %d,%f pred=%" PRId64 " \n", mpi.rank_2d, tgt_local, castUInt32ToFloat(stream[i + 1]), pred_v);
            //printf("pred_v=%u target=%u weight=%f i=%d\n", unsigned(pred_v), tgt_local, weight, i);
            assert(0 <= tgt_local && tgt_local < graph_.num_local_verts_);

            if( weight < dist[tgt_local] ) {
#if USE_DISTANCE_LOCKS
               if( !with_nq )
                  omp_set_lock(&vertices_locks_[tgt_local]);
#endif

               // todo better have a relative comparison here?
               if( comp::isLT(weight, dist[tgt_local]) ) {
                    // std::cout << " pred=" <<  pred_v <<  " update node " << graph_.invert_map_[tgt_local] << " weight:" << dist[tgt_local] << " -> " << weight << '\n';

                  //printf("...updated node %d (%f->%f, src=%d) \n", tgt_orig, std::min(100.0f, dist[tgt_orig]), weight, unsigned(pred_v));

                  if( !with_nq ) {
                     dist[tgt_local] = weight;
                     pred[tgt_local] = pred_v;
#if USE_DISTANCE_LOCKS
                     omp_unset_lock(&vertices_locks_[tgt_local]);
#endif
                  }

                  assert(comp::isLE(delta_epoch_ * delta_step_, weight)); // weight should not be in lower bucket

                  if( with_nq ) {
                     if(buf->full()) {
                        nq_.push(buf); buf = nq_empty_buffer_.get();
                     }
                     buf->append_nocheck(tgt_local, pred_v, weight);
                  }
               }
#if USE_DISTANCE_LOCKS
               else if( !with_nq )  {
                  omp_unset_lock(&vertices_locks_[tgt_local]);
               }
#endif
            }
			}
		}
		tlb->cur_buffer = buf;
		PROF(recv_proc_thread_time_ += tk_all);
	}

	//-------------------------------------------------------------//
	// bottom-up search
	//-------------------------------------------------------------//

#ifdef USE_BOTTOM_UP

	void botto_up_print_stt(int64_t num_blocks, int64_t num_vertexes, int* nq_count) {
		int64_t send_stt[2] = { num_vertexes, num_blocks };
		int64_t sum_stt[2];
		int64_t max_stt[2];
		MPI_Reduce(send_stt, sum_stt, 2, MpiTypeOf<int64_t>::type, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(send_stt, max_stt, 2, MpiTypeOf<int64_t>::type, MPI_MAX, 0, MPI_COMM_WORLD);
		if(mpi.isMaster() && sum_stt[0] != 0) {
			print_with_prefix("Bottom-Up using List. Total %f M Vertexes / %f M Blocks = %f Max %f %%+ Vertexes %f %%+ Blocks",
					to_mega(sum_stt[0]), to_mega(sum_stt[1]), to_mega(sum_stt[0]) / to_mega(sum_stt[1]),
					diff_percent(max_stt[0], sum_stt[0], mpi.size_2d),
					diff_percent(max_stt[1], sum_stt[1], mpi.size_2d));
		}
		int count_length = mpi.size_2dc;
		int start_proc = mpi.rank_2dc;
		int size_mask = count_length - 1;
		int64_t phase_count[count_length];
		int64_t phase_recv[count_length];
		for(int i = 0; i < count_length; ++i) {
			phase_count[i] = nq_count[(start_proc + i) & size_mask];
		}
		MPI_Reduce(phase_count, phase_recv, count_length, MpiTypeOf<int64_t>::type, MPI_SUM, 0, MPI_COMM_WORLD);
		if(mpi.isMaster()) {
			int64_t total_nq = 0;
			for(int i = 0; i < count_length; ++i) {
				total_nq += phase_recv[i];
			}
			print_with_prefix("Bottom-Up: %" PRId64 " vertexes found. Break down ...", total_nq);
			for(int i = 0; i < count_length; ++i) {
				print_with_prefix("step %d / %d  %f M Vertexes ( %f %% )",
						i+1, count_length, to_mega(phase_recv[i]), (double)phase_recv[i] / (double)total_nq * 100.0);
			}
		}
	}

	struct UnvisitedBitmapFunctor {
		BitmapType* visited;
		UnvisitedBitmapFunctor(BitmapType* visited__)
			: visited(visited__) { }
		BitmapType operator()(int i) { return ~(visited[i]); }
	};

	void swap_visited_memory(bool prev_bitmap_or_list) {
		// ----> Now, new_visited_ has current VIS
		if(bitmap_or_list_) { // bitmap ?
			// Since the VIS bitmap is modified in the search function,
			// we copy the VIS to the working memory to avoid corrupting the VIS bitmap.
			// bottom_up_bitmap search function begins with work_buf_ that contains current VIS bitmap.
			int64_t bitmap_width = get_bitmap_size_local();
			memory::copy_mt(work_buf_, new_visited_, bitmap_width*sizeof(BitmapType));
		}
		std::swap(new_visited_, old_visited_);
		// ----> Now, old_visited_ has current VIS
		if(!bitmap_or_list_) { // list ?
			if(prev_bitmap_or_list) { // previous level is performed with bitmap ?
				// make list from bitmap
				int step_bitmap_width = get_bitmap_size_local() / BU_SUBSTEP;
				BitmapType* bmp_vis_p[4]; get_visited_pointers(bmp_vis_p, 4, old_visited_, BU_SUBSTEP);
				TwodVertex* list_vis_p[4]; get_visited_pointers(list_vis_p, 4, new_visited_, BU_SUBSTEP);
				//int64_t shifted_c = int64_t(mpi.rank_2dc) << graph_.lgl_;
				for(int i = 0; i < BU_SUBSTEP; ++i)
					new_visited_list_size_[i] = make_list_from_bitmap(
							false, 0, UnvisitedBitmapFunctor(bmp_vis_p[i]),
							step_bitmap_width, list_vis_p[i]);
				std::swap(new_visited_, old_visited_);
				// ----> Now, old_visited_ has current VIS in the list format.
			}
			for(int i = 0; i < BU_SUBSTEP; ++i)
				std::swap(new_visited_list_size_[i], old_visited_list_size_[i]);
		}
	}

	void flush_bottom_up_send_buffer(LocalPacket* buffer, int target_rank) {
		TRACER(flush);
		int bulk_send_size = BottomUpCommHandler::BUF_SIZE;
		for(int offset = 0; offset < buffer->length; offset += bulk_send_size) {
			int length = std::min(buffer->length - offset, bulk_send_size);
			bu_comm_.put(buffer->data.b + offset, length, target_rank);
		}
		buffer->length = 0;
	}
#if BFELL
	// returns the number of vertices found in this step.
	int bottom_up_search_bitmap_process_step(
			BitmapType* phase_bitmap,
			TwodVertex phase_bmp_off,
			TwodVertex half_bitmap_width)
	{assert(0 && "see original code"); }

	// returns the number of vertices found in this step.
	TwodVertex bottom_up_search_list_process_step(
#if VERBOSE_MODE
			int64_t& num_blocks,
#endif
			TwodVertex* phase_list,
			TwodVertex phase_size,
			int8_t* vertex_enabled,
			TwodVertex* write_list,
			TwodVertex phase_bmp_off,
			TwodVertex half_bitmap_width,
			int* th_offset)
	{ assert(0 && "see original code"); }
#else // #if BFELL

	void bottom_up_search_bitmap_process_block(
			BitmapType* __restrict__ phase_bitmap,
			int off_start,
			int off_end,
			int phase_bmp_off,
			LocalPacket* buffer)
	{
		VERBOSE(int tmp_edge_relax = 0);

		int lgl = graph_.local_bits_;
		TwodVertex L = graph_.num_local_verts_;
		int r_bits = graph_.r_bits_;
		int orig_lgl = graph_.orig_local_bits_;

		const BitmapType* __restrict__ row_bitmap = graph_.row_bitmap_;
		const BitmapType* __restrict__ shared_visited = shared_visited_;
		const TwodVertex* __restrict__ row_sums = graph_.row_sums_;
		const int64_t* __restrict__ isolated_edges = NULL; // todo!
		const int64_t* __restrict__ row_starts = graph_.row_starts_;
		const LocalVertex* __restrict__ orig_vertexes = graph_.orig_vertexes_;
		const int64_t* __restrict__ edge_array = graph_.edge_array_;

		//TwodVertex lmask = (TwodVertex(1) << lgl) - 1;
		int num_send = 0;
#if CONSOLIDATE_IFE_PROC
		for(int64_t blk_bmp_off = off_start; blk_bmp_off < off_end; ++blk_bmp_off) {
			BitmapType row_bmp_i = *(row_bitmap + phase_bmp_off + blk_bmp_off);
			BitmapType visited_i = *(phase_bitmap + blk_bmp_off);
			TwodVertex bmp_row_sums = *(row_sums + phase_bmp_off + blk_bmp_off);
			BitmapType bit_flags = (~visited_i) & row_bmp_i;
			while(bit_flags != BitmapType(0)) {
				BitmapType vis_bit = bit_flags & (-bit_flags);
				BitmapType mask = vis_bit - 1;
				bit_flags &= ~vis_bit;
				TwodVertex non_zero_idx = bmp_row_sums + __builtin_popcountl(row_bmp_i & mask);
				LocalVertex tgt_orig = orig_vertexes[non_zero_idx];
				// short cut
				int64_t src = isolated_edges[non_zero_idx];
				TwodVertex bit_idx = SeparatedId(SeparatedId(src).low(r_bits + lgl)).compact(lgl, L);
				if(shared_visited[bit_idx >> PRM::LOG_NBPE] & (BitmapType(1) << (bit_idx & PRM::NBPE_MASK))) {
					// add to next queue
					visited_i |= vis_bit;
					buffer->data.b[num_send++] = ((src >> lgl) << orig_lgl) | tgt_orig;
					// end this row
					VERBOSE(tmp_edge_relax += 1);
					continue;
				}
				int64_t e_start = row_starts[non_zero_idx];
				int64_t e_end = row_starts[non_zero_idx+1];
				for(int64_t e = e_start; e < e_end; ++e) {
					int64_t src = edge_array[e];
					TwodVertex bit_idx = SeparatedId(SeparatedId(src).low(r_bits + lgl)).compact(lgl, L);
					if(shared_visited[bit_idx >> PRM::LOG_NBPE] & (BitmapType(1) << (bit_idx & PRM::NBPE_MASK))) {
						// add to next queue
						visited_i |= vis_bit;
						buffer->data.b[num_send++] = ((src >> lgl) << orig_lgl) | tgt_orig;
						// end this row
						VERBOSE(tmp_edge_relax += e - e_start + 1);
						break;
					}
				}
			} // while(bit_flags != BitmapType(0)) {
			// write back
			*(phase_bitmap + blk_bmp_off) = visited_i;
		} // #pragma omp for

#else // #if CONSOLIDATE_IFE_PROC
		for(int64_t blk_bmp_off = off_start; blk_bmp_off < off_end; ++blk_bmp_off) {
			BitmapType row_bmp_i = *(row_bitmap + phase_bmp_off + blk_bmp_off);
			BitmapType visited_i = *(phase_bitmap + blk_bmp_off);
			TwodVertex bmp_row_sums = *(row_sums + phase_bmp_off + blk_bmp_off);
			BitmapType bit_flags = (~visited_i) & row_bmp_i;
			while(bit_flags != BitmapType(0)) {
				BitmapType vis_bit = bit_flags & (-bit_flags);
				BitmapType mask = vis_bit - 1;
				bit_flags &= ~vis_bit;
				int idx = __builtin_popcountl(mask);
				TwodVertex non_zero_idx = bmp_row_sums + __builtin_popcountl(row_bmp_i & mask);
				// short cut
				TwodVertex separated_src = isolated_edges[non_zero_idx];
				TwodVertex bit_idx = (separated_src >> lgl) * L + (separated_src & lmask);
				if(shared_visited[bit_idx >> PRM::LOG_NBPE] & (BitmapType(1) << (bit_idx & PRM::NBPE_MASK))) {
					// add to next queue
					visited_i |= vis_bit;
					buffer->data.b[num_send+0] = separated_src;
					buffer->data.b[num_send+1] = (phase_bmp_off + blk_bmp_off) * PRM::NBPE + idx;
					num_send += 2;
				}
			} // while(bit_flags != BitmapType(0)) {
			// write back
			*(phase_bitmap + blk_bmp_off) = visited_i;
		} // #pragma omp for

		for(int64_t blk_bmp_off = off_start; blk_bmp_off < off_end; ++blk_bmp_off) {
			BitmapType row_bmp_i = *(row_bitmap + phase_bmp_off + blk_bmp_off);
			BitmapType visited_i = *(phase_bitmap + blk_bmp_off);
			TwodVertex bmp_row_sums = *(row_sums + phase_bmp_off + blk_bmp_off);
			BitmapType bit_flags = (~visited_i) & row_bmp_i;
			while(bit_flags != BitmapType(0)) {
				BitmapType vis_bit = bit_flags & (-bit_flags);
				BitmapType mask = vis_bit - 1;
				bit_flags &= ~vis_bit;
				int idx = __builtin_popcountl(mask);
				TwodVertex non_zero_idx = bmp_row_sums + __builtin_popcountl(row_bmp_i & mask);
				int64_t e_start = row_starts[non_zero_idx];
				int64_t e_end = row_starts[non_zero_idx+1];
				for(int64_t e = e_start; e < e_end; ++e) {
					TwodVertex separated_src = edge_array[e];
					TwodVertex bit_idx = (separated_src >> lgl) * L + (separated_src & lmask);
					if(shared_visited[bit_idx >> PRM::LOG_NBPE] & (BitmapType(1) << (bit_idx & PRM::NBPE_MASK))) {
						// add to next queue
						visited_i |= vis_bit;
						buffer->data.b[num_send+0] = separated_src;
						buffer->data.b[num_send+1] = (phase_bmp_off + blk_bmp_off) * PRM::NBPE + idx;
						num_send += 2;
						// end this row
						break;
					}
				}
			} // while(bit_flags != BitmapType(0)) {
			// write back
			*(phase_bitmap + blk_bmp_off) = visited_i;
		} // #pragma omp for
#endif // #if CONSOLIDATE_IFE_PROC

		buffer->length = num_send;
		VERBOSE(__sync_fetch_and_add(&num_edge_bottom_up_, tmp_edge_relax));
	}

	// returns the number of vertices found in this step.
	int bottom_up_search_bitmap_process_step(
			BottomUpSubstepData& data,
			int step_bitmap_width,
			volatile int* process_counter,
			int target_rank)
	{
		USER_START(bu_bmp_step);

		BitmapType* phase_bitmap = (BitmapType*)data.data;
		int phase_bmp_off = data.tag.region_id * step_bitmap_width;
		//TwodVertex L = graph_.num_local_verts_;
		//TwodVertex phase_vertex_off = L / BU_SUBSTEP * (data.tag.region_id % BU_SUBSTEP);
		ThreadLocalBuffer* tlb = thread_local_buffer_[omp_get_thread_num()];
		LocalPacket* buffer = tlb->fold_packet;

		PROF(profiling::TimeKeeper tk_all);
		PROF(profiling::TimeSpan ts_commit);

#ifndef NDEBUG
		assert (buffer->length == 0);
		assert (*process_counter == 0);
		thread_sync_.barrier();
#endif
		int tid = omp_get_thread_num();

#if 1 // dynamic partitioning
		int visited_count = 0;
		int block_width = step_bitmap_width / 40 + 1;
		while(true) {
			int off_start = __sync_fetch_and_add(process_counter, block_width);
			if(off_start >= step_bitmap_width) break; // finish
			int off_end = std::min(step_bitmap_width, off_start + block_width);

			bottom_up_search_bitmap_process_block(phase_bitmap,
					off_start, off_end, phase_bmp_off, buffer);
			PROF(extract_edge_time_ += tk_all);

			if(tid == 0) {
				// process async communication
				bottom_up_substep_->probe();
			}

			visited_count += buffer->length;
			flush_bottom_up_send_buffer(buffer, target_rank);

			PROF(commit_time_ += tk_all);
		}
#else // static partitioning
		const int num_threads = omp_get_num_threads();
		const int width_per_thread = (step_bitmap_width + num_threads - 1) / num_threads;
		const int off_start = std::min(step_bitmap_width, width_per_thread * tid);
		const int off_end = std::min(step_bitmap_width, off_start + width_per_thread);

		bottom_up_search_bitmap_process_block(phase_bitmap,
				off_start, off_end, phase_bmp_off, buffer);
		PROF(extract_edge_time_ += tk_all);

		int visited_count = buffer->length / 2;
		flush_bottom_up_send_buffer(buffer, target_rank);

		PROF(commit_time_ += tk_all);
#endif
		USER_END(bu_bmp_step);

		thread_sync_.barrier();

		PROF(extract_thread_wait_ += tk_all);
		return visited_count;
	}

	// returns the number of vertices found in this step.
	TwodVertex bottom_up_search_list_process_step(
#if VERBOSE_MODE
			int64_t& num_blocks,
#endif
			BottomUpSubstepData& data,
			int target_rank,
			int8_t* vertex_enabled,
			TwodVertex* write_list,
			TwodVertex step_bitmap_width,
			int* th_offset)
	{
		TRACER(bu_list_step);
		int max_threads = omp_get_num_threads();
		TwodVertex* phase_list = (TwodVertex*)data.data;
		TwodVertex phase_bmp_off = data.tag.region_id * step_bitmap_width;

		int lgl = graph_.local_bits_;
		int r_bits = graph_.r_bits_;
		TwodVertex L = graph_.num_local_verts_;
		int orig_lgl = graph_.orig_local_bits_;
		//TwodVertex phase_vertex_off = L / BU_SUBSTEP * (data.tag.region_id % BU_SUBSTEP);
		VERBOSE(int tmp_num_blocks = 0);
		VERBOSE(int tmp_edge_relax = 0);
		PROF(profiling::TimeKeeper tk_all);
		PROF(profiling::TimeSpan ts_commit);
		int tid = omp_get_thread_num();
		ThreadLocalBuffer* tlb = thread_local_buffer_[tid];
		LocalPacket* buffer = tlb->fold_packet;
		assert (buffer->length == 0);
		int num_send = 0;

		int64_t begin, end;
		get_partition(data.tag.length, phase_list, LOG_BFELL_SORT, max_threads, tid, begin, end);
		int num_enabled = end - begin;
		USER_START(bu_list_proc);
		for(int i = begin; i < end; ) {
			TwodVertex blk_idx = phase_list[i] >> LOG_BFELL_SORT;
			TwodVertex* phase_row_sums = graph_.row_sums_ + phase_bmp_off;
			BitmapType* phase_row_bitmap = graph_.row_bitmap_ + phase_bmp_off;
			VERBOSE(tmp_num_blocks++);

			do {
				vertex_enabled[i] = 1;
				TwodVertex tgt = phase_list[i];
				TwodVertex word_idx = tgt >> LOG_NBPE;
				int bit_idx = tgt & NBPE_MASK;
				BitmapType vis_bit = (BitmapType(1) << bit_idx);
				BitmapType row_bitmap_i = phase_row_bitmap[word_idx];
				if(row_bitmap_i & vis_bit) { // I have edges for this vertex ?
					TwodVertex non_zero_idx = phase_row_sums[word_idx] +
							__builtin_popcountl(row_bitmap_i & (vis_bit-1));
					LocalVertex tgt_orig = graph_.orig_vertexes_[non_zero_idx];
#if ISOLATE_FIRST_EDGE
					int64_t src = graph_.isolated_edges_[non_zero_idx];
					TwodVertex bit_idx = SeparatedId(SeparatedId(src).low(r_bits + lgl)).compact(lgl, L);
					if(shared_visited_[bit_idx >> LOG_NBPE] & (BitmapType(1) << (bit_idx & NBPE_MASK))) {
						// add to next queue
						vertex_enabled[i] = 0; --num_enabled;
						buffer->data.b[num_send++] = ((src >> lgl) << orig_lgl) | tgt_orig;
						// end this row
						VERBOSE(tmp_edge_relax += 1);
						continue;
					}
#endif // #if ISOLATE_FIRST_EDGE
					int64_t e_start = graph_.row_starts_[non_zero_idx];
					int64_t e_end = graph_.row_starts_[non_zero_idx+1];
					for(int64_t e = e_start; e < e_end; ++e) {
						int64_t src = graph_.edge_array_[e];
						TwodVertex bit_idx = SeparatedId(SeparatedId(src).low(r_bits + lgl)).compact(lgl, L);
						if(shared_visited_[bit_idx >> LOG_NBPE] & (BitmapType(1) << (bit_idx & NBPE_MASK))) {
							// add to next queue
							vertex_enabled[i] = 0; --num_enabled;
							buffer->data.b[num_send++] = ((src >> lgl) << orig_lgl) | tgt_orig;
							// end this row
							VERBOSE(tmp_edge_relax += e - e_start + 1);
							break;
						}
					}
				} // if(row_bitmap_i & vis_bit) {
			} while((phase_list[++i] >> LOG_BFELL_SORT) == blk_idx);
			assert(i <= end);
		}
		buffer->length = num_send;
		th_offset[tid+1] = num_enabled;
		PROF(extract_edge_time_ += tk_all);
		USER_END(bu_list_proc);
		thread_sync_.barrier();
		PROF(extract_thread_wait_ += tk_all);
#pragma omp master
		{
			TRACER(bu_list_single);
			th_offset[0] = 0;
			for(int i = 0; i < max_threads; ++i) {
				th_offset[i+1] += th_offset[i];
			}
			assert (th_offset[max_threads] <= int(data.tag.length));
		}
		thread_sync_.barrier();

		USER_START(bu_list_write);
		// make new list to send
		int offset = th_offset[tid];

		for(int i = begin; i < end; ++i) {
			if(vertex_enabled[i]) {
				write_list[offset++] = phase_list[i];
			}
		}

		USER_END(bu_list_write);
		flush_bottom_up_send_buffer(buffer, target_rank);
		PROF(commit_time_ += tk_all);
		VERBOSE(__sync_fetch_and_add(&num_blocks, tmp_num_blocks));
		VERBOSE(__sync_fetch_and_add(&num_edge_bottom_up_, tmp_edge_relax));
		thread_sync_.barrier();
		return data.tag.length - th_offset[max_threads];
	}
#endif // #if BFELL

	void bottom_up_gather_nq_size(int* visited_count) {
		TRACER(bu_gather_info);
		PROF(profiling::TimeKeeper tk_all);
		PROF(MPI_Barrier(mpi.comm_2d));
		PROF(fold_competion_wait_ += tk_all);
#if 1 // which one is faster ?
		int recv_count[mpi.size_2dc]; for(int i = 0; i < mpi.size_2dc; ++i) recv_count[i] = 1;
		MPI_Reduce_scatter(visited_count, &nq_size_, recv_count, MPI_INT, MPI_SUM, mpi.comm_2dr);
		//MPI_Allreduce(&nq_size_, &max_nq_size_, 1, MPI_INT, MPI_MAX, mpi.comm_2d);
		int64_t nq_size = nq_size_;
		MPI_Allreduce(&nq_size, &global_nq_size_, 1, MpiTypeOf<int64_t>::type, MPI_SUM, mpi.comm_2d);
#else
		int red_nq_size[mpi.size_2dc];
		struct {
			int nq_size;
			int max_nq_size;
			int64_t global_nq_size;
		} scatter_buffer[mpi.size_2dc], recv_nq_size;
		// gather information within the processor row
		MPI_Reduce(visited_count, red_nq_size, mpi.size_2dc, MPI_INT, MPI_SUM, 0, mpi.comm_2dr);
		if(mpi.rank_2dr == 0) {
			int max_nq_size = 0, sum_nq_size = 0;
			int64_t global_nq_size;
			for(int i = 0; i < mpi.size_2dc; ++i) {
				sum_nq_size += red_nq_size[i];
				if(max_nq_size < red_nq_size[i]) max_nq_size = red_nq_size[i];
			}
			// compute global_nq_size by allreduce within the processor column
			MPI_Allreduce(&sum_nq_size, &global_nq_size, 1, MpiTypeOf<int64_t>::type, MPI_SUM, mpi.comm_2dc);
			for(int i = 0; i < mpi.size_2dc; ++i) {
				scatter_buffer[i].nq_size = red_nq_size[i];
				scatter_buffer[i].max_nq_size = max_nq_size;
				scatter_buffer[i].global_nq_size = global_nq_size;
			}
		}
		// scatter information within the processor row
		MPI_Scatter(scatter_buffer, sizeof(recv_nq_size), MPI_BYTE,
				&recv_nq_size, sizeof(recv_nq_size), MPI_BYTE, 0, mpi.comm_2dr);
		nq_size_ = recv_nq_size.nq_size;
		max_nq_size_ = recv_nq_size.max_nq_size;
		global_nq_size_ = recv_nq_size.global_nq_size;
#endif
		PROF(gather_nq_time_ += tk_all);
	}

	void bottom_up_bmp_parallel_section(int *visited_count) {
		PROF(profiling::TimeKeeper tk_all);
		int bitmap_width = get_bitmap_size_local();
		int step_bitmap_width = bitmap_width / BU_SUBSTEP;
		assert (work_buf_size_ >= bitmap_width * PRM::BOTTOM_UP_BUFFER);
		int buffer_count = work_buf_size_ / (step_bitmap_width * sizeof(BitmapType));
		BitmapType* bitmap_buffer[buffer_count];
		get_visited_pointers(bitmap_buffer, buffer_count, work_buf_, BU_SUBSTEP);
		BitmapType *new_vis[BU_SUBSTEP];
		get_visited_pointers(new_vis, BU_SUBSTEP, new_visited_, BU_SUBSTEP);
		int comm_size = mpi.size_2dc;
		volatile int process_counter = 0;

		// since the first 4 buffer contains initial data, skip this region
		bottom_up_substep_->begin(bitmap_buffer + BU_SUBSTEP, buffer_count - BU_SUBSTEP,
				step_bitmap_width);

		BottomUpSubstepData data;
		int total_steps = (comm_size+1)*BU_SUBSTEP;

#pragma omp parallel
		{
			SET_OMP_AFFINITY;
			int tid = omp_get_thread_num();
			for(int step = 0; step < total_steps; ++step) {
				if(tid == 0) {
					if(step < BU_SUBSTEP) {
						data.data = bitmap_buffer[step];
						data.tag.length = step_bitmap_width;
						data.tag.region_id = mpi.rank_2dc * BU_SUBSTEP + step;
						data.tag.routed_count = 0;
						// route is set by communication lib
					}
					else {
						// receive data
						PROF(profiling::TimeKeeper tk_all);
						bottom_up_substep_->recv(&data);
						PROF(comm_wait_time_ += tk_all);
					}
					process_counter = 0;
				}
				thread_sync_.barrier();

				int target_rank = data.tag.region_id / BU_SUBSTEP;
				if(step >= BU_SUBSTEP && target_rank == mpi.rank_2dc) {
					// This is rounded and came here.
					BitmapType* src = (BitmapType*)data.data;
					BitmapType* dst = new_vis[data.tag.region_id % BU_SUBSTEP];
#pragma omp for
					for(int64_t i = 0; i < step_bitmap_width; ++i) {
						dst[i] = src[i];
					}
					thread_sync_.barrier();
				}
				else {
					visited_count[data.tag.routed_count + tid * comm_size] +=
							bottom_up_search_bitmap_process_step(
									data, step_bitmap_width, &process_counter, target_rank);
					if(tid == 0) {
						if(step < BU_SUBSTEP) {
							bottom_up_substep_->send_first(&data);
						}
						else {
							bottom_up_substep_->send(&data);
						}
					}
				}
			}
			thread_sync_.barrier();

		} // #pragma omp parallel
		PROF(parallel_reg_time_ += tk_all);
	}

	struct BottomUpBitmapParallelSection : public Runnable {
		ThisType* this_; int* visited_count;
		BottomUpBitmapParallelSection(ThisType* this__, int* visited_count_)
			: this_(this__), visited_count(visited_count_) { }
		virtual void run() { this_->bottom_up_bmp_parallel_section(visited_count); }
	};

	void bottom_up_search_bitmap() {
		TRACER(bu_bmp);
		int max_threads = omp_get_max_threads();
		int comm_size = mpi.size_2dc;
		int visited_count[comm_size*max_threads];
		for(int i = 0; i < comm_size*max_threads; ++i) visited_count[i] = 0;

		bu_comm_.prepare();
		bottom_up_bmp_parallel_section(visited_count);
		bu_comm_.run();

		// gather visited_count
		for(int tid = 1; tid < max_threads; ++tid)
			for(int i = 0; i < comm_size; ++i)
				visited_count[i + 0*comm_size] += visited_count[i + tid*comm_size];

		bottom_up_gather_nq_size(visited_count);
		VERBOSE(botto_up_print_stt(0, 0, visited_count));
		VERBOSE(bottom_up_substep_->print_stt());
	}

	void bottom_up_list_parallel_section(int *visited_count, int8_t* vertex_enabled,
			int64_t& num_blocks, int64_t& num_vertexes)
	{
		PROF(profiling::TimeKeeper tk_all);
		int bitmap_width = get_bitmap_size_local();
		int step_bitmap_width = bitmap_width / BU_SUBSTEP;
		assert (work_buf_size_ >= bitmap_width * PRM::BOTTOM_UP_BUFFER);
		int buffer_size = step_bitmap_width * sizeof(BitmapType) / sizeof(TwodVertex);
		int buffer_count = work_buf_size_ / (step_bitmap_width * sizeof(BitmapType));
		TwodVertex* list_buffer[buffer_count];
		get_visited_pointers(list_buffer, buffer_count, work_buf_, BU_SUBSTEP);
		TwodVertex *new_vis[BU_SUBSTEP];
		get_visited_pointers(new_vis, BU_SUBSTEP, new_visited_, BU_SUBSTEP);
		TwodVertex *old_vis[BU_SUBSTEP];
		get_visited_pointers(old_vis, BU_SUBSTEP, old_visited_, BU_SUBSTEP);
		int comm_size = mpi.size_2dc;

		// the first 5 buffers are working buffer
		bottom_up_substep_->begin(list_buffer + BU_SUBSTEP + 1, buffer_count - BU_SUBSTEP - 1,
				buffer_size);

		TwodVertex* back_buffer = list_buffer[BU_SUBSTEP];
		TwodVertex* write_buffer;
		BottomUpSubstepData data;
		TwodVertex sentinel_value = step_bitmap_width * NBPE;
		int total_steps = (comm_size+1)*BU_SUBSTEP;
		int max_threads = omp_get_max_threads();
		int th_offset[max_threads];

#pragma omp parallel
		{
			SET_OMP_AFFINITY;
			for(int step = 0; step < total_steps; ++step) {
#pragma omp master
				{
					if(step < BU_SUBSTEP) {
						data.data = old_vis[step];
						data.tag.length = old_visited_list_size_[step];
						write_buffer = list_buffer[step];
						data.tag.region_id = mpi.rank_2dc * BU_SUBSTEP + step;
						data.tag.routed_count = 0;
						// route is set by communication lib
					}
					else {
						// receive data
						PROF(profiling::TimeKeeper tk_all);
						bottom_up_substep_->recv(&data);
						PROF(comm_wait_time_ += tk_all);
						write_buffer = back_buffer;
					}
				}
				thread_sync_.barrier(); // TODO: compare with omp barrier

				int target_rank = data.tag.region_id / BU_SUBSTEP;
				TwodVertex* phase_list = (TwodVertex*)data.data;
				if(step >= BU_SUBSTEP && target_rank == mpi.rank_2dc) {
					// This is rounded and came here.
					int local_region_id = data.tag.region_id % BU_SUBSTEP;
					TwodVertex* dst = new_vis[local_region_id];
#pragma omp for
					for(int64_t i = 0; i < data.tag.length; ++i) {
						dst[i] = phase_list[i];
					}
#pragma omp master
					{
						new_visited_list_size_[local_region_id] = data.tag.length;
					}
					thread_sync_.barrier();
				}
				else {
					// write sentinel value to the last
					assert (data.tag.length < buffer_size - 1);
					phase_list[data.tag.length] = sentinel_value;
					int new_visited_cnt = bottom_up_search_list_process_step(
#if VERBOSE_MODE
						num_blocks,
#endif
						data, target_rank, vertex_enabled, write_buffer, step_bitmap_width, th_offset);

#pragma omp master
					{
						VERBOSE(num_vertexes += data.tag.length);
						visited_count[data.tag.routed_count] += new_visited_cnt;
						data.tag.length -= new_visited_cnt;

						if(step < BU_SUBSTEP) {
							data.data = write_buffer;
							bottom_up_substep_->send_first(&data);
						}
						else {
							back_buffer = (TwodVertex*)data.data;
							data.data = write_buffer;
							bottom_up_substep_->send(&data);
						}
					}
				}
			}
			thread_sync_.barrier();

		} // #pragma omp parallel
		PROF(parallel_reg_time_ += tk_all);
	}
/*
	struct BottomUpListParallelSection : public Runnable {
		ThisType* this_; int* visited_count; int8_t* vertex_enabled;
		int64_t num_blocks; int64_t num_vertexes;
		BottomUpListParallelSection(ThisType* this__, int* visited_count_, int8_t* vertex_enabled_)
			: this_(this__), visited_count(visited_count_) , vertex_enabled(vertex_enabled_)
			, num_blocks(0), num_vertexes(0) { }
		virtual void run() {
			this_->b;
		}
	};
*/
	void bottom_up_search_list() {
		TRACER(bu_list);

		int half_bitmap_width = get_bitmap_size_local() / 2;
		int buffer_size = half_bitmap_width * sizeof(BitmapType) / sizeof(TwodVertex);
		// TODO: reduce memory allocation
		int8_t* vertex_enabled = (int8_t*)cache_aligned_xcalloc(buffer_size*sizeof(int8_t));

		int comm_size = mpi.size_2dc;
		int visited_count[comm_size];
		for(int i = 0; i < comm_size; ++i) visited_count[i] = 0;

		bu_comm_.prepare();
		int64_t num_blocks; int64_t num_vertexes;
		bottom_up_list_parallel_section(visited_count, vertex_enabled, num_blocks, num_vertexes);
		bu_comm_.run();

		bottom_up_gather_nq_size(visited_count);
		VERBOSE(botto_up_print_stt(num_blocks, num_vertexes, visited_count));
		VERBOSE(bottom_up_substep_->print_stt());

		free(vertex_enabled); vertex_enabled = NULL;
	}

	struct BottomUpReceiver : public Runnable {
		BottomUpReceiver(ThisType* this_, int64_t* buffer_, int length_, int src_)
			: this_(this_), buffer_(buffer_), length_(length_), src_(src_) { }
		virtual void run() {
			TRACER(bu_recv);
			PROF(profiling::TimeKeeper tk_all);
			int P = mpi.size_2d;
			int r_bits = this_->graph_.r_bits_;
			int64_t r_mask = ((1 << r_bits) - 1);
			int orig_lgl = this_->graph_.orig_local_bits_;
			LocalVertex lmask = (LocalVertex(1) << orig_lgl) - 1;
			int64_t cshifted = src_ * mpi.size_2dr;
			int64_t levelshifted = int64_t(this_->current_phase_) << 48;
			int64_t* buffer = buffer_;
			int length = length_;
			int64_t* pred = this_->pred_;
			for(int i = 0; i < length; ++i) {
				int64_t v = buffer[i];
				int64_t pred_dst = v >> orig_lgl;
				LocalVertex tgt_local = v & lmask;
				int64_t pred_v = ((pred_dst >> r_bits) * P +
						cshifted + (pred_dst & r_mask)) | levelshifted;
				assert (this_->pred_[tgt_local] == -1);
				pred[tgt_local] = pred_v;
			}

			PROF(this_->recv_proc_thread_time_ += tk_all);
		}
		ThisType* const this_;
		int64_t* buffer_;
		int length_;
		int src_;
	};

#endif // USE_BOTTOM_UP

	static void printInformation()
	{
		if(mpi.isMaster() == false) return ;
		using namespace PRM;
		print_with_prefix("===== Settings and Parameters. ====");

#define PRINT_VAL(fmt, val) print_with_prefix(#val " = " fmt ".", val)
		PRINT_VAL("%d", NUM_SSSP_ROOTS);
		PRINT_VAL("%d", omp_get_max_threads());
		PRINT_VAL("%d", omp_get_nested());
		PRINT_VAL("%zd", sizeof(BitmapType));
		PRINT_VAL("%zd", sizeof(TwodVertex));

		PRINT_VAL("%d", NUMA_BIND);
		PRINT_VAL("%d", CPU_BIND_CHECK);
		PRINT_VAL("%d", PRINT_BINDING);
		PRINT_VAL("%d", SHARED_MEMORY);

		PRINT_VAL("%d", MPI_FUNNELED);
		PRINT_VAL("%d", OPENMP_SUB_THREAD);

		PRINT_VAL("%d", VERBOSE_MODE);
		PRINT_VAL("%d", PROFILING_MODE);
		PRINT_VAL("%d", DEBUG_PRINT);
		PRINT_VAL("%d", REPORT_GEN_RPGRESS);
		PRINT_VAL("%d", ENABLE_FJMPI_RDMA);
		PRINT_VAL("%d", ENABLE_FUJI_PROF);
		PRINT_VAL("%d", ENABLE_MY_ALLGATHER);
		PRINT_VAL("%d", ENABLE_INLINE_ATOMICS);

		PRINT_VAL("%d", BFELL);

		PRINT_VAL("%d", VERTEX_REORDERING);
		PRINT_VAL("%d", TOP_DOWN_SEND_LB);
		PRINT_VAL("%d", TOP_DOWN_RECV_LB);
		PRINT_VAL("%d", BOTTOM_UP_OVERLAP_PFS);

		PRINT_VAL("%d", CONSOLIDATE_IFE_PROC);

		PRINT_VAL("%d", PRE_EXEC_TIME);

		PRINT_VAL("%d", BACKTRACE_ON_SIGNAL);
#if BACKTRACE_ON_SIGNAL
		PRINT_VAL("%d", PRINT_BT_SIGNAL);
#endif
		PRINT_VAL("%d", PACKET_LENGTH);
		PRINT_VAL("%d", COMM_BUFFER_SIZE);
		PRINT_VAL("%d", SEND_BUFFER_LIMIT);
		PRINT_VAL("%d", BOTTOM_UP_BUFFER);
		PRINT_VAL("%d", NBPE);
		PRINT_VAL("%d", BFELL_SORT);
		PRINT_VAL("%f", DENOM_TOPDOWN_TO_BOTTOMUP);
		PRINT_VAL("%f", DEMON_BOTTOMUP_TO_TOPDOWN);
		PRINT_VAL("%f", DENOM_BITMAP_TO_LIST);

		PRINT_VAL("%d", VALIDATION_LEVEL);
		PRINT_VAL("%d", SGI_OMPLACE_BUG);
#undef PRINT_VAL

		if(NUM_SSSP_ROOTS == 64 && VALIDATION_LEVEL == 2)
			print_with_prefix("===== Benchmark Mode OK ====");
		else
			print_with_prefix("===== Non Benchmark Mode ====");
	}
#if VERBOSE_MODE
	void printTime(const char* fmt, double* sum, double* max, int idx) {
		print_with_prefix(fmt, sum[idx] / mpi.size_2d * 1000.0,
				diff_percent(max[idx], sum[idx], mpi.size_2d));
	}
	void printCounter(const char* fmt, int64_t* sum, int64_t* max, int idx) {
		print_with_prefix(fmt, to_mega(sum[idx] / mpi.size_2d),
				diff_percent(max[idx], sum[idx], mpi.size_2d));
	}
#endif
	void run_sssp(int64_t root, int64_t* pred) { }

	// members
	MpiBottomUpSubstepComm* bottom_up_substep_;
	CommBufferPool a2a_comm_buf_;
	TopDownCommHandler top_down_comm_;
#ifdef USE_BOTTOM_UP
	BottomUpCommHandler bottom_up_comm_;
   AsyncAlltoallManager bu_comm_;
#endif
	AsyncAlltoallManager td_comm_;
	ThreadLocalBuffer** thread_local_buffer_;
	memory::ConcurrentPool<QueuedVertexes> nq_empty_buffer_;
	memory::ConcurrentStack<QueuedVertexes*> nq_;

	// switch parameters
	double denom_to_bottom_up_; // alpha
	double denom_bitmap_to_list_; // gamma

	// delta between 0 and 1 (range of edge weights)
	float delta_step_;

	// cq_list_ is a pointer to work_buf_ can represent list or bitmap
	TwodVertex* cq_any_;

	// the following five are actual owners
	TwodVertex* nq_list_;
	int64_t* cq_root_list_;
	int64_t* nq_root_list_;
	float* cq_distance_list_;
	float* nq_distance_list_;
	TwodVertex cq_size_;
	int nq_size_;
	int max_nq_size_;
	int64_t global_nq_size_;

	// per local vertex
	int32_t* vertices_pos_;
#if USE_DISTANCE_LOCKS
	omp_lock_t* vertices_locks_;
#endif

	// size = local bitmap width
	// These two buffer is swapped at the beginning of every backward step
	void* new_visited_; // shared memory but point to the local portion
	void* old_visited_; // shared memory but point to the local portion
	void* visited_buffer_; // shared memory but point to the local portion
	void* visited_buffer_orig_; // shared memory
	// size = 2
	int new_visited_list_size_[BU_SUBSTEP];
	int old_visited_list_size_[BU_SUBSTEP];


	// 1. CQ at the top-down phase
	// 2.
	// 3. Working memory at the bottom-up search phase
	void* work_buf_; // buffer to be used for different purposes
	int64_t work_buf_size_; // in bytes
	enum Work_buf_state work_buf_state_;

	// length of cq distance list  NOT in bytes!
   int64_t cq_distance_buf_length_;
   // length of nq distance AND vertex list  NOT in bytes!
   int64_t nq_buf_length_;

   BitmapType* vertices_isSettled_;
   BitmapType* vertices_isSettledLocal_;
   BitmapType* vertices_isInCurrentBucket_; // marks whether local vertex is in current bucket
	BitmapType* shared_visited_; // shared memory
	TwodVertex* nq_recv_buf_; // shared memory (memory space is shared with work_buf_)

	int64_t* pred_; // in default mode: predecessor per vertex, passed from main method...in presolving mode: the current root vertex of the path
	float* dist_;  // distance per vertex, passed from main method


	int64_t* pred_presol_;
	float*  dist_presol_;

	struct SharedDataSet {
		memory::SpinBarrier *sync;
		int *offset; // max(max_threads*2+1, 2*mpi.size_z+1)
	} s_; // shared memory

	int64_t global_visited_vertices_;
	int delta_epoch_; // current epoch, i.e. iteration, for delta-stepping, starting at 0
	int current_phase_;
	int current_level_;
	bool is_light_phase_;
	bool is_bellman_ford_;
	bool forward_or_backward_;
	bool bitmap_or_list_;
	bool growing_or_shrinking_;
	bool packet_buffer_is_dirty_;
   bool next_forward_or_backward_;
   bool next_bitmap_or_list_;
   bool has_settled_vertices_;
   bool reset_root_grad1;
   bool settled_is_clean;
   bool is_presolve_mode_;
	memory::SpinBarrier thread_sync_;
	std::vector<int64_t> prev_buckets_sizes;

	VERBOSE(int64_t num_edge_top_down_);
	VERBOSE(int64_t num_td_large_edge_);
	VERBOSE(int64_t num_edge_bottom_up_);
	struct {
		void* thread_local_;
		void* shared_memory_;
	} buffer_;
	PROF(profiling::TimeSpan extract_edge_time_);
	PROF(profiling::TimeSpan isolated_edge_time_);
	PROF(profiling::TimeSpan extract_thread_wait_);
	PROF(profiling::TimeSpan parallel_reg_time_);
	PROF(profiling::TimeSpan seq_proc_time_);
	PROF(profiling::TimeSpan commit_time_);
	PROF(profiling::TimeSpan comm_wait_time_);
	PROF(profiling::TimeSpan fold_competion_wait_);
	PROF(profiling::TimeSpan recv_proc_thread_time_);
	PROF(profiling::TimeSpan recv_proc_thread_large_time_);
	PROF(profiling::TimeSpan gather_nq_time_);


private:
	bool bellman_ford_is_promising(void) const;
	void initialize_sssp_run();
	void execute_sssp_run(int64_t root);
   void finalize_sssp_run(int64_t root);
   void run_sssp_phases();
   void initialize_next_epoch(int64_t root, bool& hasNewEpoch);
   void initialize_heavy_phase(bool& epochHasHeavyEdges);
};


// is switch to Bellman-Ford promising?
bool SsspBase::bellman_ford_is_promising(void) const {
   const size_t n_prev_buckets = prev_buckets_sizes.size();
     // todo have some better number here!
   if( n_prev_buckets < 4 )
      return false;

   const int64_t bsize_curr = prev_buckets_sizes[n_prev_buckets - 1];
   const int64_t bsize_max = *std::max_element(prev_buckets_sizes.begin(), prev_buckets_sizes.end());

   // todo add some proper relative parameter
   const int64_t min_max = std::min<int64_t>(graph_.num_global_verts_ / 100, 1000);
   if( bsize_max < min_max )
      return false;

   // todo at least have some parameter here
   const int64_t lower_bound = (bsize_max * BELLMAN_FORD_SWITCH_RATIO);

   if( bsize_curr < lower_bound )
      return true;

   return false;
}


// initializes
void SsspBase::initialize_sssp_run()
{
   settled_is_clean = false;
   has_settled_vertices_ = false;
   is_bellman_ford_ = false;
   next_forward_or_backward_ = true; // begin with forward search
   next_bitmap_or_list_ = false;
   global_visited_vertices_ = 1; // count the root vertex
   current_level_ = 0;
   delta_epoch_ = 0;
   max_nq_size_ = 1;
   global_nq_size_ = 0;
   forward_or_backward_ = next_forward_or_backward_;
   bitmap_or_list_ = next_bitmap_or_list_;
   growing_or_shrinking_ = true;
   prev_buckets_sizes.clear();

   const int64_t num_local_verts = graph_.num_local_verts_;
   const int64_t bitmap_width = get_bitmap_size_local();
   int64_t* const restrict pred = pred_;
   float* const restrict dist = dist_;

#ifndef NDEBUG
   for( int64_t i = 0; i < num_local_verts; ++i )
         assert(vertices_pos_[i] == -1);
#endif

   memory::clean_mt(vertices_isSettledLocal_, bitmap_width * sizeof(*vertices_isSettledLocal_));

#pragma omp parallel
   {
#pragma omp for nowait
      for(int64_t i = 0; i < num_local_verts; ++i)
         pred[i] = -1;

#pragma omp for nowait
      for(int64_t i = 0; i < num_local_verts; ++i)
         dist[i] = std::numeric_limits<float>::max();
   }

   assert (nq_.stack_.size() == 0);
   assert(!mpi.isYdimAvailable());
   //if(mpi.isYdimAvailable()) s_.sync->barrier();
}


// actually execute the computations
void SsspBase::execute_sssp_run(int64_t root)
{
   // main loop, iterates over the epochs
   while( true ) {
      bool hasNewEpoch;

      // start new epoch (with light phase or Bellman-Ford)
      initialize_next_epoch(root, hasNewEpoch);

      if( !hasNewEpoch )
         break;

      run_sssp_phases();
      clear_nq_stack();

      // NOTE: in case of Bellman-Ford, the last phase finished everything
      if( is_bellman_ford_ ) {
         if( mpi.isMaster() )
            std::cout << "FINISHED SSSP computation (number of Bellman-Ford iterations: " << current_phase_ << ")\n";
         break;
      }

      // start heavy phase
      bool epochHasHeavyEdges = false;
      initialize_heavy_phase(epochHasHeavyEdges);

      if( epochHasHeavyEdges ) {
         run_sssp_phases();
         clear_nq_stack();
      }
   }

   clear_nq_stack();
}



// finalizes a single computation: updates the distances and predecessors
void SsspBase::finalize_sssp_run(int64_t root) {

   const int64_t num_orig_local_verts = graph_.num_orig_local_verts_;
   const int64_t num_local_verts = graph_.num_local_verts_;
   const LocalVertex* const restrict invert_map = graph_.invert_map_;

   assert(work_buf_size_ >= int64_t(sizeof(int64_t)) * graph_.num_orig_local_verts_);
   assert(num_local_verts <= num_orig_local_verts);

   if( reset_root_grad1 ) {
      const int64_t root_local = vertex_local(root);
      const int64_t reordered = graph_.reorder_map_[root_local];
      assert(!graph_.local_vertex_isDeg1(reordered));
      assert(pred_[reordered] == root);

      const int64_t word_idx = reordered >> LOG_NBPE;
      const int64_t bit_idx = reordered & NBPE_MASK;
      graph_.is_grad1_bitmap_[word_idx] |= BitmapType(1) << bit_idx;

      assert(graph_.local_vertex_isDeg1(reordered));
   }

   // 1. update the predecessors

   int64_t* const restrict pred = pred_;
   int64_t* pred_tmp = (int64_t*) work_buf_;
   assert(work_buf_size_ % alignof(*pred_tmp) == 0);

#pragma omp parallel for schedule(static)
   for( int64_t i = 0; i < num_orig_local_verts; i++ ) {
      pred_tmp[i] = -1;
   }

#pragma omp parallel for schedule(static)
   for( int64_t i = 0; i < num_local_verts; i++ ) {
      if( pred[i] < 0 ) {
         assert(pred[i] == -1);
         continue;
      }
      const LocalVertex tgt_orig = invert_map[i];
      assert(0 <= tgt_orig && tgt_orig < num_orig_local_verts);
      pred_tmp[tgt_orig] = pred[i];
   }
   memory::copy_mt(pred, pred_tmp, num_orig_local_verts * sizeof(*pred_tmp));


   // 2. update the distances

   float* const restrict dist = dist_;
   float* dist_tmp = (float*) work_buf_;
   work_buf_state_ = Work_buf_state::dists;
   assert(work_buf_size_ % alignof(*dist_tmp) == 0);

#pragma omp parallel for schedule(static)
   for( int64_t i = 0; i < num_orig_local_verts; i++ ) {
      dist_tmp[i] = std::numeric_limits<float>::max();
   }

#pragma omp parallel for schedule(static)
   for( int64_t i = 0; i < num_local_verts; i++ ) {
      if( dist[i] >= comp::infinity )
         continue;
      const LocalVertex tgt_orig = invert_map[i];
      assert(0 <= tgt_orig && tgt_orig < num_orig_local_verts);
      dist_tmp[tgt_orig] = dist[i];
   }
   memory::copy_mt(dist, dist_tmp, num_orig_local_verts * sizeof(*dist_tmp));
}

// initializes next epoch of delta-stepping algorithm
void SsspBase::initialize_next_epoch(int64_t root, bool& hasNewEpoch)
{
   int64_t global_next_bucket_size = 0;
   const bool isFirstIteration = (prev_buckets_sizes.size() == 0);

   current_phase_ = 0;
   is_light_phase_ = true;
   hasNewEpoch = false;

   if( isFirstIteration ) {
      if( is_presolve_mode_ ) {
         assert(root == std::numeric_limits<int64_t>::max());
      }
      else {
         first_expand(root);
      }
      hasNewEpoch = true;
      global_next_bucket_size = 1;
   }
   else {
      if( bellman_ford_is_promising() ) {
         if( is_presolve_mode_ && 1 ) {
            int todo; // try to deactivate again, maybe to expensive? And little benefit? ...better make more roots!
            hasNewEpoch = false;
            return;
         }

         is_bellman_ford_ = true;
         next_bitmap_or_list_ = true;

         if( mpi.isMaster() )
            printf("Switched to Bellman-Ford! \n");

         if( !has_settled_vertices_ )
            bucket_expand_settled_bitmap(graph_.num_global_verts_);

         ++delta_epoch_;
         top_down_expand_bucket(global_next_bucket_size);
      }
      else {
         const int index = bucket_get_next_nonempty(false);
         assert(index > delta_epoch_);

         if( index == std::numeric_limits<int>::max() ) {
            global_next_bucket_size = 0;
         }
         else {
            delta_epoch_ = index;
            top_down_expand_bucket(global_next_bucket_size);
         }
      }

      hasNewEpoch = (global_next_bucket_size > 0);
   }

   prev_buckets_sizes.push_back(global_next_bucket_size);
   //memset(vertices_isInCurrentBucket_, 0, get_bitmap_size_local() * sizeof(*vertices_isInCurrentBucket_));
}


// initializes heavy phase of delta-stepping algorithm
void SsspBase::initialize_heavy_phase(bool& epochHasHeavyEdges)
{
   assert(is_light_phase_);
   is_light_phase_ = false;
   const int64_t global_nq_size = bucket_get_nq_size();
   epochHasHeavyEdges = (global_nq_size > 0);

   // todo have some relative limit for global_next_bucket_size_!
   if( (global_nq_size >= 1000 || has_settled_vertices_ ) )
   {
      bucket_expand_settled_bitmap(global_nq_size);
      assert(has_settled_vertices_);
   }

   if( !epochHasHeavyEdges )
      return;

   int64_t size;
   top_down_expand_bucket(size);
   assert(global_nq_size == size);
}


// one phase(s) of delta-stepping algorithm (heavy or light); can also do Bellman-Ford if parameter is_bellman_ford_ is set
void SsspBase::run_sssp_phases()
{
#if VERBOSE_MODE
   using namespace profiling;
#endif

   while(true) {
      ++current_phase_;
      ++current_level_;
      assert(current_level_ < std::numeric_limits<uint16_t>::max());

#if VERBOSE_MODE
      double prev_time = MPI_Wtime();
      num_edge_top_down_ = num_td_large_edge_ = num_edge_bottom_up_ = 0;
#endif
#if ENABLE_FUJI_PROF
      fapp_start(prof_mes[(int)forward_or_backward_], 0, 0);
      start_collection(prof_mes[(int)forward_or_backward_]);
#endif
      TRACER(level);

      // search phase //
      forward_or_backward_ = next_forward_or_backward_;
      bitmap_or_list_ = next_bitmap_or_list_;
      assert(forward_or_backward_ && "backward currently not implemented");

      top_down_search();

      global_visited_vertices_ += global_nq_size_;

#if VERBOSE_MODE
      const double cur_fold_time = MPI_Wtime() - prev_time;
      assert(cur_fold_time >= 0.0);
      fold_time += cur_fold_time;
      total_edge_top_down += num_edge_top_down_;
      total_edge_bottom_up += num_edge_bottom_up_;
#if PROFILING_MODE
      AsyncAlltoallManager* a2a_comm = forward_or_backward_ ? &td_comm_ : NULL;

      if(forward_or_backward_) {
         extract_edge_time_.submit("forward edge", current_phase_);
      }
      else {
#if ISOLATE_FIRST_EDGE
         isolated_edge_time_.submit("isolated edge", current_phase_);
#endif
         extract_edge_time_.submit("backward edge", current_phase_);
      }
      extract_thread_wait_.submit("extract thread wait", current_phase_);

      parallel_reg_time_.submit("parallel region", current_phase_);
      commit_time_.submit("extract commit", current_phase_);
      if(!forward_or_backward_) { // backward
         comm_wait_time_.submit("bottom-up communication wait", current_phase_);
      }
      fold_competion_wait_.submit("fold completion wait", current_phase_);
      recv_proc_thread_time_.submit("recv proc thread", current_phase_);
      if(forward_or_backward_) {
         recv_proc_thread_large_time_.submit("recv proc thread large", current_phase_);
      }
      gather_nq_time_.submit("gather NQ info", current_phase_);
      seq_proc_time_.submit("sequential processing", current_phase_);
      a2a_comm->submit_prof_info(current_phase_, forward_or_backward_);

      if(forward_or_backward_) {
         profiling::g_pis.submitCounter(num_edge_top_down_, "top-down edge relax", current_phase_);
         profiling::g_pis.submitCounter(num_td_large_edge_, "top-down large edge", current_phase_);
      }
      else
         profiling::g_pis.submitCounter(num_edge_bottom_up_, "bottom-up edge relax", current_phase_);

      int64_t send_num_edges[] = { num_edge_top_down_, num_td_large_edge_, num_edge_bottom_up_ };
      int64_t recv_num_edges[3];
      MPI_Reduce(send_num_edges, recv_num_edges, 3, MpiTypeOf<int64_t>::type, MPI_SUM, 0, MPI_COMM_WORLD);
      num_edge_top_down_ = recv_num_edges[0];
      num_td_large_edge_ = recv_num_edges[1];
      num_edge_bottom_up_ = recv_num_edges[2];
#endif // #if PROFILING_MODE
      prev_time = MPI_Wtime();
#endif // #if VERBOSE_MODE
#if ENABLE_FUJI_PROF
      stop_collection(prof_mes[(int)forward_or_backward_]);
      fapp_stop(prof_mes[(int)forward_or_backward_], 0, 0);
      fapp_start("expand", 0, 0);
      start_collection("expand");
#endif

      // set direction (bottom-up or top-down) of next iteration //
      next_bitmap_or_list_ = false;//!forward_or_backward_;

      if( global_nq_size_ == 0 )
         break;

      top_down_expand();
      clear_nq_stack();

#if ENABLE_FUJI_PROF
      stop_collection("expand");
      fapp_stop("expand", 0, 0);
#endif
#if VERBOSE_MODE
      const double cur_expand_time = MPI_Wtime() - prev_time;
      assert(cur_expand_time >= 0.0);
      expand_time += cur_expand_time;
#endif
   }
}


void SsspBase::run_sssp(int64_t root, int64_t* pred, float* dist)
{
	SET_AFFINITY;
#if ENABLE_FUJI_PROF
	fapp_start("initialize", 0, 0);
	start_collection("initialize");
#endif
	TRACER(run_sssp);
	pred_ = pred;
	dist_ = dist;
	is_presolve_mode_ = false;
	assert(!cq_root_list_ && !nq_root_list_);

#if VERBOSE_MODE
	using namespace profiling;

	const double start_time = MPI_Wtime();
	expand_settled_bitmap_time = expand_buckets_time = expand_time = fold_time = 0.0;
	total_edge_top_down = total_edge_bottom_up = 0;
	g_tp_comm = g_bu_pred_comm = g_bu_bitmap_comm = g_bu_list_comm = g_expand_bitmap_comm = g_expand_list_comm = 0;
#endif

	initialize_sssp_run();
   assert(prev_buckets_sizes.size() == 0);

#if VERBOSE_MODE
	if(mpi.isMaster()) print_with_prefix("Time of initialize: %f ms", (MPI_Wtime() - start_time) * 1000.0);
#endif

#if ENABLE_FUJI_PROF
	stop_collection("initialize");
	fapp_stop("initialize", 0, 0);
	char *prof_mes[] = { "bottom-up", "top-down" };
#endif


	execute_sssp_run(root);

	finalize_sssp_run(root);

#if VERBOSE_MODE
	if(mpi.isMaster()) print_with_prefix("Time of SSSP: %f ms", (MPI_Wtime() - start_time) * 1000.0);
   const int time_cnt = 4;
	double send_time[] = { fold_time, expand_time, expand_buckets_time, expand_settled_bitmap_time };
	double sum_time[time_cnt], max_time[time_cnt];
	MPI_Reduce(send_time, sum_time, time_cnt, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(send_time, max_time, time_cnt, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   if(mpi.isMaster()) {
      printTime("Avg time of fold: %f ms, %f %%+", sum_time, max_time, 0);
      printTime("Avg time of expand: %f ms, %f %%+", sum_time, max_time, 1);
      printTime("Avg time of bucket expand: %f ms, %f %%+", sum_time, max_time, 2);
      printTime("Avg time of bitmap expand: %f ms, %f %%+", sum_time, max_time, 3);
   }

#if 0
   int64_t total_edge_relax = total_edge_top_down + total_edge_bottom_up;
   int cnt_cnt = 9;
   int64_t send_cnt[] = { g_tp_comm, g_bu_pred_comm, g_bu_bitmap_comm,
         g_bu_list_comm, g_expand_bitmap_comm, g_expand_list_comm,
         total_edge_top_down, total_edge_bottom_up, total_edge_relax };
   int64_t sum_cnt[cnt_cnt], max_cnt[cnt_cnt];
	MPI_Reduce(send_cnt, sum_cnt, cnt_cnt, MpiTypeOf<int64_t>::type, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(send_cnt, max_cnt, cnt_cnt, MpiTypeOf<int64_t>::type, MPI_MAX, 0, MPI_COMM_WORLD);

	if(mpi.isMaster()) {
		printCounter("Avg top-down fold recv: %f MiB, %f %%+", sum_cnt, max_cnt, 0);
		printCounter("Avg bottom-up pred update recv: %f MiB, %f %%+", sum_cnt, max_cnt, 1);
		printCounter("Avg bottom-up bitmap send: %f MiB, %f %%+", sum_cnt, max_cnt, 2);
		printCounter("Avg bottom-up list send: %f MiB, %f %%+", sum_cnt, max_cnt, 3);
		printCounter("Avg expand bitmap recv: %f MiB, %f %%+", sum_cnt, max_cnt, 4);
		printCounter("Avg expand list recv: %f MiB, %f %%+", sum_cnt, max_cnt, 5);
		printCounter("Avg top-down traversed edges: %f MiB, %f %%+", sum_cnt, max_cnt, 6);
		printCounter("Avg bottom-up traversed edges: %f MiB, %f %%+", sum_cnt, max_cnt, 7);
		printCounter("Avg total relaxed traversed: %f MiB, %f %%+", sum_cnt, max_cnt, 8);
	}
#endif
#endif
}
#undef debug

#endif /* SSSP_SRC_SSSP_HPP_ */

/*
 * main.cc
 *
 *  Created on: Mar 8, 2022
 *      Author: Daniel Rehfeldt
 */


/*
 * main.cc
 *
 *  Created on: Dec 9, 2011
 *      Author: koji
 * 
 */

// C includes
#include <kagen.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <getopt.h>

// C++ includes
#include <string>
#include <iostream>

#include "parameters.h"
#include "utils.hpp"
#include "primitives.hpp"
#include "../src/generator/graph_generator.hpp"
#include "../src/sssp/graph_constructor.hpp"
#include "../src/sssp/validate.hpp"
#include "../src/sssp/benchmark_helper.hpp"
#include "../src/sssp/sssp.hpp"
#include "../src/sssp/sssp_presol.hpp"

static
void run_graph500sssp(int SCALE, int edgefactor, std::string const& kagen_option_string = "")
{
	using namespace PRM;
	SET_AFFINITY;

	double sssp_times[64], validate_times[64], edge_counts[64];
	LogFileFormat log = {0};
	int root_start = read_log_file(&log, SCALE, edgefactor, sssp_times, validate_times, edge_counts);
	if(mpi.isMaster() && root_start != 0)
		print_with_prefix("Resume from %d th run", root_start);

	const char* filepath = getenv("TMPFILE");
	EdgeListStorage<WeightedEdge, 8*1024*1024> edge_list(
//	EdgeListStorage<UnweightedPackedEdge, 512*1024> edge_list(
			(int64_t(1) << SCALE) * edgefactor / mpi.size_2d, filepath);

	SsspBase::printInformation();

	if(mpi.isMaster()) print_with_prefix("Graph generation");
	double generation_time = MPI_Wtime();
	if (kagen_option_string == "") {
	  generate_graph_spec2010(&edge_list, SCALE, edgefactor);
	} else {
	  if(mpi.isMaster()) print_with_prefix("Using Kagen with option string: %s", kagen_option_string.c_str());
	  generate_graph_kagen(&edge_list, kagen_option_string);
	}
	generation_time = MPI_Wtime() - generation_time;

	//edge_list.writeGraphToFile(("first_list" + std::to_string(mpi.rank) + ".txt").c_str());

	if(mpi.isMaster()) print_with_prefix("Graph construction");
	// Create SSSP instance and the *COMMUNICATION THREAD*.
	SsspBase sssp_instance;
	SsspPresolver sssp_presolver(sssp_instance);
	double construction_time = MPI_Wtime();
	sssp_instance.construct(&edge_list);
	construction_time = MPI_Wtime() - construction_time;

	if(mpi.isMaster()) print_with_prefix("Redistributing edge list...");
	double redistribution_time = MPI_Wtime();
	redistribute_edge_2d(&edge_list);
	redistribution_time = MPI_Wtime() - redistribution_time;

	int64_t sssp_roots[NUM_SSSP_ROOTS];
	int num_sssp_roots = NUM_SSSP_ROOTS;
	find_roots(sssp_instance.graph_, sssp_roots, num_sssp_roots);
	//sssp_roots[0] = 7; // 16463; // 1356013851; //433575;
	const int64_t max_used_vertex = find_max_used_vertex(sssp_instance.graph_);
	const int64_t nlocalverts = sssp_instance.graph_.pred_size();

	//edge_list.writeGraphToFile("graph.tgf", true, true);
	//edge_list.writeGraphToFile(("second_list" + std::to_string(mpi.rank_2d) + ".txt").c_str());
	//sssp_instance.graph_.writeLocalToFile(("graph_local" + std::to_string(mpi.rank_2d) + ".txt").c_str());

	//MPI_Barrier(MPI_COMM_WORLD);
	//MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

	// todo make this in a cleaner way
	free(sssp_instance.graph_.has_edge_bitmap_); sssp_instance.graph_.has_edge_bitmap_= nullptr;

	int64_t *pred = static_cast<int64_t*>(
		cache_aligned_xmalloc(nlocalverts*sizeof(pred[0])));

   float *dist = static_cast<float*>(
      cache_aligned_xmalloc(nlocalverts*sizeof(dist[0])));

#if INIT_PRED_ONCE	// Only Spec2010 needs this initialization
#pragma omp parallel for
	for(int64_t i = 0; i < nlocalverts; ++i) {
		pred[i] = -1;
		dist[i] = std::numeric_limits<float>::max();
	}
#endif

	bool result_ok = true;

	if(root_start == 0)
		init_log(SCALE, edgefactor, generation_time, construction_time, redistribution_time, &log);

	sssp_instance.prepare_sssp();

		double time_left = PRE_EXEC_TIME;
        for(int c = root_start; time_left > 0.0; ++c) {
                if(mpi.isMaster())  print_with_prefix("========== Pre Running SSSP %d ==========", c);
                MPI_Barrier(mpi.comm_2d);
                double sssp_time = MPI_Wtime();
                sssp_instance.run_sssp(sssp_roots[c % num_sssp_roots], pred, dist);
                sssp_time = MPI_Wtime() - sssp_time;
                if(mpi.isMaster()) {
                        print_with_prefix("Time for SSSP %d is %f", c, sssp_time);
                        time_left -= sssp_time;
                }
               MPI_Bcast(&time_left, 1, MPI_DOUBLE, 0, mpi.comm_2d);
        }
	bool do_presol = true;
	const char* presol_time_char = std::getenv("PRESOL_SECONDS");

	if( presol_time_char ) {
	  if (std::stoi(presol_time_char) == 0) {
	    do_presol=false;
	  }
	}

	if (do_presol){
	  const int niterations = 4000;
	  sssp_presolver.presolve_sssp(niterations, pred, dist);

	  MPI_Barrier(mpi.comm_2d);
	  if(mpi.isMaster()) print_with_prefix("Preproc is finished \n");
	  num_sssp_roots = NUM_SSSP_ROOTS;
	  find_roots_presolved(sssp_instance.graph_, sssp_roots, num_sssp_roots);
	  //MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}


/////////////////////
	for(int i = root_start; i < num_sssp_roots; ++i) {
	//for(int i = 0; i < num_bfs_roots; ++i) {
	   VERBOSE_MEM(print_max_memory_usage());

		if(mpi.isMaster()) {
		   print_with_prefix("========== Running SSSP %d ==========", i);
		   printf("Root=%" PRId64 " \n", sssp_roots[i]);
		}

#if ENABLE_FUJI_PROF
		fapp_start("bfs", i, 1);
#endif
		MPI_Barrier(mpi.comm_2d);
#if ENABLE_FJMPI_PROF
                FJMPI_Collection_start();
#endif
		PROF(profiling::g_pis.reset());
		sssp_times[i] = MPI_Wtime();
		sssp_instance.run_sssp(sssp_roots[i], pred, dist);
		sssp_times[i] = MPI_Wtime() - sssp_times[i];
#if ENABLE_FJMPI_PROF
                FJMPI_Collection_stop();
#endif
#if ENABLE_FUJI_PROF
		fapp_stop("bfs", i, 1);
#endif
		PROF(profiling::g_pis.printResult());
		if(mpi.isMaster()) {
			print_with_prefix("Time for SSSP %d is %f", i, sssp_times[i]);
			print_with_prefix("Validating SSSP %d", i);
		}

		validate_times[i] = MPI_Wtime();
		int64_t edge_visit_count = 0;
#if VALIDATION_LEVEL >= 2
		result_ok = validate_sssp_result(&edge_list, dist, max_used_vertex + 1, nlocalverts, sssp_roots[i], pred, &edge_visit_count);
#elif VALIDATION_LEVEL == 1
		if(i == 0) {
			result_ok = validate_sssp_result(
						&edge_list, max_used_vertex + 1, nlocalverts, sssp_roots[i], pred, &edge_visit_count);
			pf_nedge[SCALE] = edge_visit_count;
		}
		else {
			edge_visit_count = pf_nedge[SCALE];
		}
#else
		edge_visit_count = pf_nedge[SCALE];
#endif
		validate_times[i] = MPI_Wtime() - validate_times[i];
		edge_counts[i] = (double)edge_visit_count;

		if(mpi.isMaster()) {
			print_with_prefix("Validate time for SSSP %d is %f", i, validate_times[i]);
			print_with_prefix("Number of traversed edges is %" PRId64 "", edge_visit_count);
			print_with_prefix("TEPS for SSSP %d is %g", i, edge_visit_count / sssp_times[i]);
		}

		if(result_ok == false) {
			break;
		}

		update_log_file(&log, sssp_times[i], validate_times[i], edge_visit_count);
	}
	sssp_instance.end_sssp();

	if(mpi.isMaster()) {
	  fprintf(stdout, "============= Result ==============\n");
	  fprintf(stdout, "SCALE:                          %d\n", SCALE);
	  fprintf(stdout, "edgefactor:                     %d\n", edgefactor);
	  fprintf(stdout, "NSSSPs:                         %d\n", num_sssp_roots);
	  fprintf(stdout, "graph_generation:               %g\n", generation_time);
	  fprintf(stdout, "num_mpi_processes:              %d\n", mpi.size_2d);
	  fprintf(stdout, "construction_time:              %g\n", construction_time);
	  fprintf(stdout, "redistribution_time:            %g\n", redistribution_time);
	  print_sssp_result(num_sssp_roots, sssp_times, validate_times, edge_counts, result_ok);
	}

#if ENABLE_FJMPI_PROF
	FJMPI_Collection_print("Communication Statistics\n");
#endif
	
   free(dist);
	free(pred);
}


int main(int argc, char** argv)
{
  // if option --kagen_option_string is passed, use its, value, else process the args normally
  std::string kagen_option_string = "";
  int c;
  int digit_optind = 0;
  while(true) {
    
    static struct option long_options[] = {
      {"kagen_option_string", required_argument, 0, 0},
      {0, 0, 0, 0}
    };
    c = getopt_long(argc, argv, "", long_options, &digit_optind);
    if (c == -1)
      break;
    switch (c) {
    case 0:
      kagen_option_string = optarg;
      break;
    default:
      break;
    }
  }
  int SCALE = 16;
  int edgefactor = 16; // nedges / nvertices, i.e., 2*avg. degree
  if (kagen_option_string == "") {
    // Parse arguments.

    if (argc >= 2) SCALE = atoi(argv[1]);
    if (argc >= 3) edgefactor = atoi(argv[2]);
    if (argc <= 1 || argc >= 4 || SCALE == 0 || edgefactor == 0) {
      fprintf(IMD_OUT, "Usage: %s SCALE edgefactor\n"
	      "SCALE = log_2(# vertices) [integer, required]\n"
	      "edgefactor = (# edges) / (# vertices) = .5 * (average vertex degree) [integer, defaults to 16]\n"
	      "(Random number seed are in main.c)\n",
	      argv[0]);
      return 0;
    }
  }

  setup_globals(argc, argv, SCALE, edgefactor);

  run_graph500sssp(SCALE, edgefactor, kagen_option_string);

  cleanup_globals();
  return 0;
}

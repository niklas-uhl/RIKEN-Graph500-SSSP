add_executable(sssp-parallel main.cc)

# if( "${MY_SYSTEM}" STREQUAL "Fugaku" ) target_link_libraries(sssp-parallel
# PUBLIC ) else()
target_link_libraries(sssp-parallel PUBLIC OpenMP::OpenMP_CXX MPI::MPI_CXX)
target_link_libraries(sssp-parallel PRIVATE KaGen::KaGen)
# endif()

# todo make user flags!
# target_compile_definitions(sssp-parallel PRIVATE SCOREP=false VERBOSE_MODE=0
#                                                  VERTEX_REORDERING=2)
if(RIKEN_GRAPH500_USE_SCOREP)
  target_compile_definitions(sssp-parallel PRIVATE "SCOREP=true")
else()
  target_compile_definitions(sssp-parallel PRIVATE "SCOREP=false")
endif()
if(RIKEN_GRAPH500_VERBOSE)
  target_compile_definitions(sssp-parallel PRIVATE VERBOSE_MODE=0)
else()
  target_compile_definitions(sssp-parallel PRIVATE VERBOSE_MODE=1)
endif()
target_compile_definitions(sssp-parallel PRIVATE "-DVERTEX_REORDERING=${RIKEN_VERTEX_REORDERING_MODE}")

target_compile_features(sssp-parallel PRIVATE cxx_std_14)

target_link_libraries(sssp-parallel PRIVATE generator sssp utils)

add_executable(sssp-sequential main-sequential.cc)

target_link_libraries(sssp-sequential PRIVATE sssp-seq)
target_compile_features(sssp-sequential PRIVATE cxx_std_14)

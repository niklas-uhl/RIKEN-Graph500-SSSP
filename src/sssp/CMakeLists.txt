add_library(sssp low_level_func.cc)

target_link_libraries(sssp PRIVATE utils MPI::MPI_CXX OpenMP::OpenMP_CXX)

target_include_directories(sssp PUBLIC ${CMAKE_CURRENT_LIST_DIR})

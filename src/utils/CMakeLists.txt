add_library(utils INTERFACE)

target_include_directories(utils INTERFACE ${CMAKE_CURRENT_LIST_DIR})

target_link_libraries(utils INTERFACE MPI::MPI_CXX OpenMP::OpenMP_CXX)
target_compile_features(utils INTERFACE cxx_std_14)

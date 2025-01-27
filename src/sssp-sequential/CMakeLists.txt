add_library(sssp-seq csrgraph.cc shortestpath.cc)

target_include_directories(sssp-seq PUBLIC ${CMAKE_CURRENT_LIST_DIR})

target_compile_features(sssp-seq PUBLIC cxx_std_14)

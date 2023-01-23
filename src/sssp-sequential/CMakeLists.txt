add_library(sssp-seq 
	csrgraph.cc
	shortestpath.cc
)

target_include_directories(sssp-seq
    PUBLIC
    "${PROJECT_SOURCE_DIR}/src/sssp-sequential/"
)

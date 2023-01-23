add_executable(sssp-parallel
    main.cc
)

if( "${MY_SYSTEM}" STREQUAL "Fugaku" )
    target_link_libraries(sssp-parallel
        PUBLIC
    )
else()
    target_link_libraries(sssp-parallel
        PUBLIC
        OpenMP::OpenMP_CXX
    )
endif()

# todo make user flags!
target_compile_definitions(sssp-parallel 
    PRIVATE 
    SCOREP=false
    VERBOSE_MODE=0
    VERTEX_REORDERING=2 
)

target_link_libraries(sssp-parallel
    PRIVATE
    generator
    sssp
    utils
)



add_executable(sssp-sequential
    main-sequential.cc
)

target_link_libraries(sssp-sequential
    PRIVATE
    sssp-seq
)

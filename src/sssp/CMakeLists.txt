add_library(sssp
    low_level_func.cc
)

target_link_libraries(sssp
    PRIVATE
    utils
)

target_include_directories(sssp
    PUBLIC
    "${PROJECT_SOURCE_DIR}/src/sssp/"
)

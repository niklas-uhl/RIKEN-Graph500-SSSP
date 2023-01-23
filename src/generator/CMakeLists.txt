add_library(generator
    splittable_mrg.c
)


target_link_libraries(generator
    PRIVATE
    utils
)

target_include_directories(generator
    PUBLIC
    "${PROJECT_SOURCE_DIR}/src/generator/"
)

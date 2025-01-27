add_library(generator splittable_mrg.c)

target_include_directories(generator PUBLIC ${CMAKE_CURRENT_LIST_DIR})

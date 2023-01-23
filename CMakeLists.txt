cmake_minimum_required(VERSION 3.10)

project(Graph500-SSSP)

set(CMAKE_CXX_STANDARD 14)
#set(ADD_C_AND_CXX_FLAGS  -Wall -pedantic  -W -Wno-unknown-warning-option -Wno-unknown-warning -Wpointer-arith -Wcast-align -Wwrite-strings -Wshadow  -Wredundant-decls -Wdisabled-optimization -Wno-long-long -Wno-unknown-pragmas -Wno-unused-parameter

set(CMAKE_MODULE_PATH "${CMAKE_BINARY_DIR}" "${CMAKE_MODULE_PATH}")
set(CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}" "${CMAKE_PREFIX_PATH}")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

set(MY_SYSTEM "Default" CACHE STRING "System to run on")
set(GCC_BASE -Drestrict=__restrict__ -D__STDC_CONSTANT_MACROS -D__STDC_LIMIT_MACROS -D__STDC_FORMAT_MACROS -ffast-math -msse4.2)

# make 'Release' the default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)
#find_package(MPI REQUIRED)
# todo do that in a proper way! Should not be hard-set

if( "${MY_SYSTEM}" STREQUAL "Fugaku" )
    message(STATUS "system=${MY_SYSTEM}; using Fugaku")
    
    set(CMAKE_CXX_COMPILER mpiFCCpx)
    set(CMAKE_C_COMPILER  mpifccpx)
    set(CMAKE_CXX_FLAGS -Kopenmp)
else()
    message(STATUS "system=${MY_SYSTEM}; using default")

    set(CMAKE_CXX_COMPILER mpicxx)
    set(CMAKE_C_COMPILER  mpicc)
endif()

add_compile_options(-Wall)

# todo
if (1)
    add_compile_options(${GCC_BASE})
endif()


option(BUILD_TESTS "Build tests" OFF)
enable_testing()

add_subdirectory(apps)
add_subdirectory(src)

if (BUILD_TESTS)
    add_subdirectory(tests)
endif ()


message(STATUS "BUILD TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_CXX_COMPILER_VERSION ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")


cmake_minimum_required(VERSION 3.10)

project(Graph500-SSSP)

set(CMAKE_CXX_STANDARD 14)

option(RIKEN_GRAPH500_BUILD_TESTS "Build tests" OFF)
option(RIKEN_GRAPH500_USE_SCOREP "Use ScoreP" OFF)
option(RIKEN_GRAPH500_VERBOSE "Verbose output" OFF)
set(RIKEN_VERTEX_REORDERING_MODE "2" CACHE STRING "Vertex reordering mode")
#set(ADD_C_AND_CXX_FLAGS  -Wall -pedantic  -W -Wno-unknown-warning-option -Wno-unknown-warning -Wpointer-arith -Wcast-align -Wwrite-strings -Wshadow  -Wredundant-decls -Wdisabled-optimization -Wno-long-long -Wno-unknown-pragmas -Wno-unused-parameter

# set(CMAKE_MODULE_PATH "${CMAKE_BINARY_DIR}" "${CMAKE_MODULE_PATH}")
# set(CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}" "${CMAKE_PREFIX_PATH}")

# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

set(MY_SYSTEM "Default" CACHE STRING "System to run on")
set(GCC_BASE -Drestrict=__restrict__ -D__STDC_CONSTANT_MACROS -D__STDC_LIMIT_MACROS -D__STDC_FORMAT_MACROS -ffast-math -msse4.2)

include(FetchContent)

set(KAGEN_BUILD_APPS OFF)
set(KAGEN_BUILD_EXAMPLES OFF)
set(KAGEN_NODEPS ON)
FetchContent_Declare(kagen
  GIT_REPOSITORY https://github.com/KarlsruheGraphGeneration/KaGen
  GIT_TAG 49dea30
  SYSTEM
)

FetchContent_MakeAvailable(kagen)
target_compile_features(kagen PUBLIC cxx_std_20)

# # make 'Release' the default build type
# if(NOT CMAKE_BUILD_TYPE)
#     set(CMAKE_BUILD_TYPE Release)
# endif()

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

# todo do that in a proper way! Should not be hard-set

# if( "${MY_SYSTEM}" STREQUAL "Fugaku" )
#     message(STATUS "system=${MY_SYSTEM}; using Fugaku")
    
#     set(CMAKE_CXX_COMPILER mpiFCCpx)
#     set(CMAKE_C_COMPILER  mpifccpx)
#     set(CMAKE_CXX_FLAGS -Kopenmp)
# else()
#     message(STATUS "system=${MY_SYSTEM}; using default")

#     set(CMAKE_CXX_COMPILER mpicxx)
#     set(CMAKE_C_COMPILER  mpicc)
# endif()

# todo
if (1)
    # add_compile_options(${GCC_BASE})
endif()

add_subdirectory(apps)
add_subdirectory(src)

if (RIKEN_GRAPH500_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif ()


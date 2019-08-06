#Gabriele's CMake for Mac OS 10.15-Beta Catalina
cmake_minimum_required(VERSION 3.14)
project(EPECsolve)
add_compile_options(-std=c++11)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_SHARED_LINKER_FLAGS "-lgurobi_c++ -lgurobi81, -lblas -llapack -lm -lpthread")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

#From https://github.com/LiangliangNan/PolyFit/blob/master/cmake/FindGUROBI.cmake
if (NOT GUROBI_FOUND)
    # Hardcoded search paths
    set(SEARCH_PATHS_FOR_HEADERS
            "$ENV{GUROBI_HOME}/include"
            "/Library/gurobi811/mac64/include"
            "C:\\dev\\gurobi811\\win64\\include"
            )

    set(SEARCH_PATHS_FOR_LIBRARIES
            "$ENV{GUROBI_HOME}/lib"
            "/Library/gurobi811/mac64/lib"
            "C:\\dev\\gurobi811\\win64\\lib"
            )

    find_path(GUROBI_INCLUDE_DIR gurobi_c++.h
            PATHS ${SEARCH_PATHS_FOR_HEADERS}
            )


    find_library( GUROBI_C_LIBRARY
            NAMES gurobi81
            PATHS ${SEARCH_PATHS_FOR_LIBRARIES}
            )

    find_library( GUROBI_CXX_LIBRARY_DEBUG
            NAMES gurobi_c++ gurobi_c++mdd2017
            PATHS ${SEARCH_PATHS_FOR_LIBRARIES}
            )

    find_library( GUROBI_CXX_LIBRARY_RELEASE
            NAMES gurobi_c++ gurobi_c++md2017
            PATHS ${SEARCH_PATHS_FOR_LIBRARIES}
            )

    # setup header file directories
    set(GUROBI_INCLUDE_DIRS ${GUROBI_INCLUDE_DIR})

    # setup libraries files
    set(GUROBI_LIBRARIES ${GUROBI_C_LIBRARY}
            debug ${GUROBI_CXX_LIBRARY_DEBUG}
            optimized ${GUROBI_CXX_LIBRARY_RELEASE})

endif()

# Check that Gurobi was successfully found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GUROBI DEFAULT_MSG GUROBI_INCLUDE_DIRS)
find_package_handle_standard_args(GUROBI DEFAULT_MSG GUROBI_LIBRARIES)
mark_as_advanced(GUROBI_LIBRARIES GUROBI_INCLUDE_DIRS GUROBI_INCLUDE_DIR)
include_directories( ${GUROBI_INCLUDE_DIR} )
include_directories( "/usr/local/boost-1.70.0/include" )
include_directories( "src" )
include_directories( "/usr/local/include" )
add_executable(EPEC
               src/EPEC.cpp
               src/epecsolve.h
               src/Games.cpp
               src/games.h
               src/LCPtoLP.cpp
               src/lcptolp.h
               src/Models.cpp
               src/models.h)
add_executable(EPECtest
               src/Games.cpp
               src/games.h
               src/LCPtoLP.cpp
               src/lcptolp.h
               src/Models.cpp
               src/models.h
               test/EPEC.cpp)

target_link_libraries( EPEC ${GUROBI_LIBRARIES})
target_link_libraries( EPECtest /usr/local/boost-1.70.0/lib/libboost_unit_test_framework.a ${GUROBI_LIBRARIES})

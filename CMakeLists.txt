############################################
################ CONFIGURATION #############
#### Here yoy can edit with your paths #####
############################################
if (ROSSOBIANCO)
    set(BOOST_ROOT "/home/x86_64-unknown-linux_ol7-gnu/boost-1.70.0")
    set(ARMA_PATH "/home/draggabr/.local/armadillo")
    set(ARMA_LIB "${ARMA_PATH}/lib64/libarmadillo.so")
ELSE ()
    set(BOOST_ROOT "/usr/local/boost-1.70.0/")
    set(ARMA_PATH "/usr/local")
    set(ARMA_LIB "/usr/local/lib/libarmadillo.dylib")
ENDIF ()
set(GUROBI_PATH_LINUX "/home/gurobi/8.1.0/linux64")
set(GUROBI_PATH_MAC "/Library/gurobi811/mac64")
set(GUROBI_PATH_WINDOWS "C:\\dev\\gurobi811\\win64")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})


############################################


############################################
################ DO NOT MODIFY #############
############################################
cmake_minimum_required(VERSION 3.10)
project(EPECsolve)
add_compile_options(-std=c++11)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_SHARED_LINKER_FLAGS "-lgurobi_c++ -lgurobi81 -lm -lpthread -O2 -lopenblas -llapack -lpthread")

#From https://github.com/LiangliangNan/PolyFit/blob/master/cmake/FindGUROBI.cmake
if (NOT GUROBI_FOUND)
    # Hardcoded search paths
    set(SEARCH_PATHS_FOR_HEADERS
        "$ENV{GUROBI_HOME}/include"
        "${GUROBI_PATH_MAC}/include"
        "${GUROBI_PATH_LINUX}/include"
        "${GUROBI_PATH_WINDOWS}/include"
        )

    set(SEARCH_PATHS_FOR_LIBRARIES
        "$ENV{GUROBI_HOME}/lib"
		"/opt/gurobi811/linux64/lib"
        "${GUROBI_PATH_MAC}/lib"
        "${GUROBI_PATH_LINUX}/lib"
        "${GUROBI_PATH_WINDOWS}/lib"
        )

    find_path(GUROBI_INCLUDE_DIR gurobi_c++.h
              PATHS ${SEARCH_PATHS_FOR_HEADERS}
              )


    find_library(GUROBI_C_LIBRARY
                 NAMES gurobi81
                 PATHS ${SEARCH_PATHS_FOR_LIBRARIES}
                 )

    find_library(GUROBI_CXX_LIBRARY_DEBUG
                 NAMES gurobi_c++ gurobi_c++mdd2017
                 PATHS ${SEARCH_PATHS_FOR_LIBRARIES}
                 )

    find_library(GUROBI_CXX_LIBRARY_RELEASE
                 NAMES gurobi_c++ gurobi_c++md2017
                 PATHS ${SEARCH_PATHS_FOR_LIBRARIES}
                 )

    # setup header file directories
    set(GUROBI_INCLUDE_DIRS  ${GUROBI_INCLUDE_DIR})

    # setup libraries files
    set(GUROBI_LIBRARIES ${GUROBI_C_LIBRARY}
        debug ${GUROBI_CXX_LIBRARY_DEBUG}
        optimized ${GUROBI_CXX_LIBRARY_RELEASE})

endif ()
include(FindPackageHandleStandardArgs)
SET(Boost_USE_STATIC_LIBS ON)
find_package(Boost 1.70.0 REQUIRED COMPONENTS program_options unit_test_framework log log_setup)
find_package_handle_standard_args(GUROBI DEFAULT_MSG GUROBI_INCLUDE_DIRS)
find_package_handle_standard_args(GUROBI DEFAULT_MSG GUROBI_LIBRARIES)
mark_as_advanced(GUROBI_LIBRARIES GUROBI_INCLUDE_DIRS GUROBI_INCLUDE_DIR)
include_directories(${GUROBI_INCLUDE_DIR})
include_directories(${BOOST_ROOT}/include)
include_directories(${ARMA_PATH}/include)
include_directories("src")
include_directories("include")
add_executable(EPEC
               src/EPEC.cpp
               src/epecsolve.h
               src/Games.cpp
               src/games.h
               src/LCPtoLP.cpp
               src/lcptolp.h
               src/utils.h
               src/Utils.cpp
               src/Models.cpp
               src/models.h)
add_executable(EPECInstances
               src/makeInstances.cpp
               src/epecsolve.h
               src/Games.cpp
               src/games.h
               src/LCPtoLP.cpp
               src/lcptolp.h
               src/utils.h
               src/Utils.cpp
               src/Models.cpp
               src/models.h)
add_executable(EPECtest
               src/Games.cpp
               src/games.h
               src/LCPtoLP.cpp
               src/lcptolp.h
               src/utils.h
               src/Utils.cpp
               src/Models.cpp
               src/models.h
               test/EPEC.cpp)
target_link_libraries(EPECInstances Boost::program_options Boost::log Boost::log_setup ${GUROBI_LIBRARIES} ${ARMA_LIB})
target_link_libraries(EPEC Boost::program_options Boost::log Boost::log_setup ${GUROBI_LIBRARIES} ${ARMA_LIB})
target_link_libraries(EPECtest Boost::program_options Boost::unit_test_framework Boost::log Boost::log_setup ${GUROBI_LIBRARIES} ${ARMA_LIB})
message("\n\nEPEC Cmake configuration terminated:")
message("\tWorking directory: ${CMAKE_CURRENT_SOURCE_DIR}")
message("\tArmadillo (lib): ${ARMA_PATH}/include")
message("\tArmadillo (include): ${ARMA_LIB}")
message("\tGurobi (lib): ${GUROBI_LIBRARIES}")
message("\tGurobi (include): ${GUROBI_INCLUDE_DIRS}")
message("\tBoost (include): ${BOOST_ROOT}/include")
############################################

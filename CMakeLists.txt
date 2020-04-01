############################################
################ CONFIGURATION #############
#### Here you can edit with your paths #####
############################################
if (ROSSOBIANCO)
    set(BOOST_ROOT "/home/x86_64-unknown-linux_ol7-gnu/boost-1.70.0")
    set(ARMA_PATH "/home/draggabr/.local/armadillo")
    set(ARMA_LIB "${ARMA_PATH}/lib64/libarmadillo.so")
ELSE ()
    set(BOOST_ROOT "/usr/local/boost-1.70.0/")
    set(ARMA_PATH "/usr/local")
    set(ARMA_LIB "/usr/local/lib/libarmadillo.dylib")
    set(Boost_USE_MULTITHREADED TRUE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++11-narrowing -Wdefaulted-function-deleted")
ENDIF ()
set(GUROBI_PATH_LINUX "/home/gurobi/9.0.0/linux64")
set(GUROBI_PATH_MAC "/Library/gurobi900/mac64")
set(GUROBI_PATH_WINDOWS "C:\\dev\\gurobi900\\win64")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})



############################################
################ DO NOT MODIFY #############
############################################
cmake_minimum_required(VERSION 3.10)
project(EPECsolve)
add_compile_options(-std=c++11)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_SHARED_LINKER_FLAGS "-lgurobi_c++ -lgurobi90 -lm -O2 -lopenblas -llapack -lpthread")

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
        "/opt/gurobi900/linux64/lib"
        "${GUROBI_PATH_MAC}/lib"
        "${GUROBI_PATH_LINUX}/lib"
        "${GUROBI_PATH_WINDOWS}/lib"
        )

    find_path(GUROBI_INCLUDE_DIR gurobi_c++.h
              PATHS ${SEARCH_PATHS_FOR_HEADERS}
              )


    find_library(GUROBI_C_LIBRARY
                 NAMES gurobi90
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
    set(GUROBI_INCLUDE_DIRS ${GUROBI_INCLUDE_DIR})

    # setup libraries files
    set(GUROBI_LIBRARIES ${GUROBI_C_LIBRARY}
        debug ${GUROBI_CXX_LIBRARY_DEBUG}
        optimized ${GUROBI_CXX_LIBRARY_RELEASE})

endif ()
include(FindPackageHandleStandardArgs)
SET(Boost_USE_STATIC_LIBS ON)
if (ROSSOBIANCO)
    find_package(Boost 1.70.0 REQUIRED COMPONENTS program_options unit_test_framework log log_setup)
else ()
    find_package(Boost 1.72.0 REQUIRED COMPONENTS program_options unit_test_framework log log_setup)
endif ()
find_package_handle_standard_args(GUROBI DEFAULT_MSG GUROBI_INCLUDE_DIRS)
find_package_handle_standard_args(GUROBI DEFAULT_MSG GUROBI_LIBRARIES)
mark_as_advanced(GUROBI_LIBRARIES GUROBI_INCLUDE_DIRS GUROBI_INCLUDE_DIR)
include_directories(${GUROBI_INCLUDE_DIR})
include_directories(${BOOST_ROOT}/include)
include_directories(${ARMA_PATH}/include)
include_directories("src")
include_directories("include")
add_library(EPECSolve SHARED
            src/epecsolve.h
            src/Games.cpp
            src/games.h
            src/LCP/LCP.cpp
            src/LCP/lcp.h
            src/LCP/PolyLCP.cpp
            src/LCP/polylcp.h
            src/utils.h
            src/Utils.cpp
            src/models.h
            src/Models.cpp
            src/algorithms/algorithms.h
            src/algorithms/InnerApproximation.cpp
            src/algorithms/innerapproximation.h
            src/algorithms/CombinatorialPNE.cpp
            src/algorithms/combinatorialpne.h
            src/algorithms/FullEnumeration.cpp
            src/algorithms/fullenumeration.h
            src/algorithms/OuterApproximation.cpp
            src/algorithms/outerapproximation.h
            src/LCP/OuterLCP.cpp
            src/LCP/outerlcp.h)
target_include_directories(EPECSolve PRIVATE src)

add_executable(EPEC
               src/CommandLineInterface.cpp)
add_executable(EPECInstances
               instances/makeInstances.cpp)
add_executable(EPECChileArgentina
               instances/ChileArgentina.cpp)
add_executable(EPECExample
               examples/example.cpp)
add_executable(EPECtest
               test/EPEC.cpp)
target_link_libraries(EPECInstances EPECSolve Boost::program_options Boost::log Boost::log_setup ${GUROBI_LIBRARIES} ${ARMA_LIB})
target_link_libraries(EPECChileArgentina EPECSolve Boost::program_options Boost::log Boost::log_setup ${GUROBI_LIBRARIES} ${ARMA_LIB})
target_link_libraries(EPECExample EPECSolve Boost::program_options Boost::log Boost::log_setup ${GUROBI_LIBRARIES} ${ARMA_LIB})
target_link_libraries(EPEC EPECSolve Boost::program_options Boost::log Boost::log_setup ${GUROBI_LIBRARIES} ${ARMA_LIB})
target_link_libraries(EPECtest EPECSolve Boost::program_options Boost::unit_test_framework Boost::log Boost::log_setup ${GUROBI_LIBRARIES} ${ARMA_LIB})
message("\n\nEPEC Cmake configuration terminated:")
message("\tWorking directory: ${CMAKE_CURRENT_SOURCE_DIR}")
message("\tArmadillo (lib) : ${ARMA_PATH}/include")
message("\tArmadillo (include) : ${ARMA_LIB}")
message("\tGurobi (lib) : ${GUROBI_LIBRARIES}")
message("\tGurobi (include) : ${GUROBI_INCLUDE_DIRS}")
message("\tBoost (include) : ${BOOST_ROOT}/include ")
############################################

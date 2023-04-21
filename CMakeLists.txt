project(EPECsolve)
############################################
################ CONFIGURATION #############
#### Here yoy can edit with your paths #####
############################################
if (ROSSOBIANCO)
    set(BOOST_ROOT "/home/x86_64-unknown-linux_ol7-gnu/boost-1.70.0")
ELSE ()
    set(BOOST_ROOT "/opt/homebrew/Cellar/boost/1.78.0_1")
ENDIF ()
#set(CMAKE_OSX_ARCHITECTURES "x86_64")
set(CMAKE_OSX_ARCHITECTURES "arm64")
set(GUROBI_PATH_LINUX "/home/gurobi/9.1.1/linux64")
set(GUROBI_PATH_MAC "/Library/gurobi950/macos_universal2")
set(GUROBI_PATH_WINDOWS "C:\\dev\\gurobi910\\win64")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})


############################################


############################################
################ DO NOT MODIFY #############
############################################
cmake_minimum_required(VERSION 3.10)
add_compile_options(-std=c++11)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_SHARED_LINKER_FLAGS "-lm -lpthread")

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
        "${GUROBI_PATH_MAC}/lib"
        "${GUROBI_PATH_LINUX}/lib"
        "${GUROBI_PATH_WINDOWS}/lib"
        )

    find_path(GUROBI_INCLUDE_DIR gurobi_c++.h
              PATHS ${SEARCH_PATHS_FOR_HEADERS}
              )


    find_library(GUROBI_C_LIBRARY
                 NAMES gurobi95
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
SET(Boost_USE_STATIC_LIBS OFF)
find_package(Armadillo REQUIRED)
find_package(Boost 1.70.0 REQUIRED COMPONENTS program_options unit_test_framework log log_setup)
find_package_handle_standard_args(GUROBI DEFAULT_MSG GUROBI_INCLUDE_DIRS)
find_package_handle_standard_args(GUROBI DEFAULT_MSG GUROBI_LIBRARIES)
mark_as_advanced(GUROBI_LIBRARIES GUROBI_INCLUDE_DIRS GUROBI_INCLUDE_DIR)
include_directories(${GUROBI_INCLUDE_DIR})
include_directories(${BOOST_ROOT}/include)
include_directories(${ARMADILLO_INCLUDE_DIRS})
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
add_executable(EPECChileArgentina
               src/ChileArgentina.cpp
               src/epecsolve.h
               src/Games.cpp
               src/games.h
               src/LCPtoLP.cpp
               src/lcptolp.h
               src/utils.h
               src/Utils.cpp
               src/Models.cpp
               src/models.h)
add_executable(EPECSGM
               src/SGM.cpp
               src/epecsolve.h
               src/Games.cpp
               src/games.h
               src/LCPtoLP.cpp
               src/lcptolp.h
               src/utils.h
               src/Utils.cpp
               src/Models.cpp
               src/models.h)
add_executable(EPECExample
               src/example.cpp
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

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(EPECInstances Boost::program_options Boost::log Boost::log_setup ${GUROBI_LIBRARIES} ${ARMADILLO_LIBRARIES})
target_link_libraries(EPECChileArgentina Boost::program_options Boost::log Boost::log_setup ${GUROBI_LIBRARIES} ${ARMADILLO_LIBRARIES})
target_link_libraries(EPECSGM Boost::program_options Boost::log Boost::log_setup ${GUROBI_LIBRARIES} ${ARMADILLO_LIBRARIES})
target_link_libraries(EPECExample Boost::program_options Boost::log Boost::log_setup ${GUROBI_LIBRARIES} ${ARMADILLO_LIBRARIES})
target_link_libraries(EPEC Boost::program_options Boost::log Boost::log_setup ${GUROBI_LIBRARIES} ${ARMADILLO_LIBRARIES})
target_link_libraries(EPECtest Boost::program_options Boost::unit_test_framework Boost::log Boost::log_setup ${GUROBI_LIBRARIES} ${ARMADILLO_LIBRARIES})
message("\n\nEPEC Cmake configuration terminated:")
message("\tWorking directory: ${CMAKE_CURRENT_SOURCE_DIR}")
message("\tArmadillo (lib): ${ARMADILLO_INCLUDE_DIRS}")
message("\tArmadillo (include): ${ARMADILLO_LIBRARIES}")
message("\tGurobi (lib): ${GUROBI_LIBRARIES}")
message("\tGurobi (include): ${GUROBI_INCLUDE_DIRS}")
message("\tBoost (include): ${BOOST_ROOT}/include")
############################################

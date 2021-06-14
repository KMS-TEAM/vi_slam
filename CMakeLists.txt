cmake_minimum_required(VERSION 3.0)
project(vi_slam)

include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    project(vi_slam LANGUAGES CXX CUDA)
else()
    project(vi_slam LANGUAGES CXX)
endif()

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "-std=c++14 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE  "-std=c++14 -O3 -fopenmp -pthread")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

############### dependencies ######################
# Eigen
include_directories("/usr/include/eigen3")

# OpenCV
find_package(OpenCV 4.2 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# pangolin
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# G2O
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

# ceres
find_package(Ceres REQUIRED)
include_directories( ${Sophus_INCLUDE_DIRS} )

# glog
find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

# gtest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# gflags
find_package(GFlags REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIRS})

# csparse
find_package(CSparse REQUIRED)
include_directories(${CSPARSE_INCLUDE_DIR})

# pcl
find_package( PCL REQUIRED )
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#vilib
#find_package(vilib REQUIRED)
#find_package(CUDA REQUIRED)
#message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")

set(THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        ${Sophus_LIBRARIES}
        ${Pangolin_LIBRARIES} GL GLU GLEW glut
        ${PCL_LIBRARIES}
        ${PCL_COMMON_LIBRARIES}
        ${PCL_IO_LIBRARIES}
        ${PCL_FILTERS_LIBRARIES}
        ${CERES_LIBRARIES}
        ${GTEST_BOTH_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${GFLAGS_LIBRARIES}
        ${CSPARSE_LIBRARY}
        pthread
        #vilib::vilib
        g2o_core g2o_stuff g2o_types_sba g2o_solver_csparse g2o_csparse_extension
        )

############### source and test ######################
include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/thirdparty)
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(app)

############# Lib Interface ####################




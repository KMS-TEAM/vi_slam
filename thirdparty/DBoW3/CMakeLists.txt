cmake_minimum_required(VERSION 2.8)
project(DBoW3)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 -march=native")

set(HDRS_DBOW3
  DBoW3/src/BowVector.h    
  DBoW3/src/FeatureVector.h
  DBoW3/src/ScoringObject.h   
  DBoW3/src/Database.h
  DBoW3/src/DBoW3.h
  DBoW3/src/DescManip.h
  DBoW3/src/exports.h
  DBoW3/src/QueryResults.h
  DBoW3/src/quicklz.h
  DBoW3/src/Vocabulary.h
  DBoW3/src/timers.h
  )
set(SRCS_DBOW3
  DBoW3/src/BowVector.cpp   
  DBoW3/src/FeatureVector.cpp
  DBoW3/src/ScoringObject.cpp
  DBoW3/src/Database.cpp
  DBoW3/src/DescManip.cpp
  DBoW3/src/QueryResults.cpp
  DBoW3/src/quicklz.c
  DBoW3/src/Vocabulary.cpp
  )

set(HDRS_DUTILS
  DUtils/Random.h
  DUtils/Timestamp.h
  )
set(SRCS_DUTILS
  DUtils/Random.cpp
  DUtils/Timestamp.cpp
  )

  find_package(OpenCV 4.2 REQUIRED)

  set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

  include_directories(${OpenCV_INCLUDE_DIRS})
  add_library(DBoW3 SHARED ${SRCS_DBOW3} ${SRCS_DUTILS})
  target_link_libraries(DBoW3 ${OpenCV_LIBS})


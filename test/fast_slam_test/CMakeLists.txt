
include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    project(fast-test LANGUAGES CXX CUDA)
else()
    project(fast-test LANGUAGES CXX)
endif()

find_package(OpenCV COMPONENTS core imgproc features2d highgui)
find_package(CUDA REQUIRED)

add_executable(test_fast_class test_fast_class.cpp)

target_link_libraries( test_fast_class
        vilib::vilib
        geometry
        opencv_core
        opencv_imgproc
        opencv_features2d
        opencv_highgui
        ${CUDA_LIBRARIES}
        ${THIRD_PARTY_LIBS})
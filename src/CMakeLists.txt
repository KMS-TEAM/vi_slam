file(GLOB_RECURSE SOURCES
        basics/*.cpp
        datastructures/*.cpp
        display/*.cpp

        geometry/cameramodels/camera.cpp
        geometry/cameramodels/pinhole.cpp
        geometry/cameramodels/kannalabrandt8.cpp
        geometry/epipolar_geometry.cpp
        #geometry/fast_cuda.cpp
        geometry/fextractor.cpp
        geometry/fmatcher.cpp
        geometry/motion_estimation.cpp

        optimization/*.cpp
        core/*cpp
        )

add_library(vi_slam SHARED ${SOURCES})

target_link_libraries( vi_slam
        #vilib::vilib
        ${THIRD_PARTY_LIBS}
        )


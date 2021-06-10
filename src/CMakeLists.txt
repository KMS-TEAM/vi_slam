add_library(basics SHARED
        basics/basics.cpp
        basics/config.cpp
        basics/yaml.cpp
        basics/eigen_funcs.cpp
        basics/opencv_funcs.cpp
        basics/converter.cpp
        )

add_library(display SHARED
        display/viewer.cpp
        display/display_lib.cpp
        )

add_library(geometry SHARED
        geometry/camera.cpp
        geometry/feature_match.cpp
        geometry/epipolar_geometry.cpp
        geometry/motion_estimation.cpp
        geometry/fast_cuda.cpp
        )

add_library( optimization SHARED
        optimization/g2o_ba.cpp
        )

add_library( core SHARED
        core/vo_io.cpp
        core/vo_commons.cpp
        core/vo.cpp
        core/localmapping.cpp
        core/loopclosing.cpp
        core/system.cpp
        core/tracking.cpp
        )

add_library( datastructures SHARED
        datastructures/keyframe.cpp
        datastructures/keyframedatabase.cpp
        datastructures/map.cpp
        datastructures/mappoint.cpp
        datastructures/frame.cpp
        )

target_link_libraries( basics
        ${THIRD_PARTY_LIBS}
        )

target_link_libraries( display
        ${THIRD_PARTY_LIBS} basics
        )

target_link_libraries( geometry
        vilib::vilib
        ${THIRD_PARTY_LIBS}
        basics
        )

target_link_libraries( optimization
        ${THIRD_PARTY_LIBS}
        basics
        )

target_link_libraries(datastructures
        ${THIRD_PARTY_LIBS}
        geometry
        basics
        optimization
        )

target_link_libraries( core
        ${THIRD_PARTY_LIBS}
        geometry
        basics
        optimization
        datastructures
        )


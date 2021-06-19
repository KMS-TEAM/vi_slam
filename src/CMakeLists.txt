add_library(vi_slam SHARED
        basics/basics.cpp
        basics/config.cpp
        basics/converter.cpp
        basics/eigen_funcs.cpp
        basics/opencv_funcs.cpp
        basics/yaml.cpp

        core/localmapping.cpp
        core/loopclosing.cpp
        core/monoinitializer.cpp
        core/system.cpp
        core/tracking.cpp
        core/vo_commons.cpp
        core/vo.cpp
        core/vo_io.cpp

        datastructures/frame.cpp
        datastructures/keyframe.cpp
        datastructures/keyframedatabase.cpp
        datastructures/map.cpp
        datastructures/mappoint.cpp

        display/display_lib.cpp
        display/framedrawer.cpp
        display/mapdrawer.cpp
        display/viewer.cpp

        geometry/camera.cpp
        geometry/epipolar_geometry.cpp
        geometry/fast_cuda.cpp
        geometry/fextractor.cpp
        geometry/fmatcher.cpp
        geometry/motion_estimation.cpp

        optimization/g2o_ba.cpp
        optimization/optimizer.cpp
        optimization/pnpsolver.cpp
        optimization/sim3solver.cpp)

target_link_libraries( vi_slam
        #vilib::vilib
        ${THIRD_PARTY_LIBS}
        )


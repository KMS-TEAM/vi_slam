file(GLOB_RECURSE SOURCES

        basics/basics.cpp
        basics/config.cpp
        basics/converter.cpp
        basics/eigen_funcs.cpp
        basics/opencv_funcs.cpp
        basics/yaml.cpp
        basics/logger.cpp

        datastructures/atlas.cpp
        datastructures/frame.cpp
        datastructures/imu.cpp
        datastructures/keyframe.cpp
        datastructures/keyframedatabase.cpp
        datastructures/map.cpp
        datastructures/mappoint.cpp

        display/display_lib.cpp
        display/framedrawer.cpp
        display/mapdrawer.cpp
        display/viewer.cpp

        geometry/cameramodels/camera.cpp
        geometry/cameramodels/pinhole.cpp
        geometry/cameramodels/kannalabrandt8.cpp
        geometry/epipolar_geometry.cpp
        #geometry/fast_cuda.cpp
        geometry/fextractor.cpp
        geometry/fmatcher.cpp
        geometry/motion_estimation.cpp

        optimization/g2o_ba.cpp
        optimization/g2otypes.cpp
        optimization/mlpnpsolver.cpp
        optimization/optimizer.cpp
        optimization/optimizetypes.cpp
        optimization/pnpsolver.cpp
        optimization/sim3solver.cpp
        optimization/gtsamoptimizer.cpp

        core/core_commons.cpp
        core/localmapping.cpp
        core/loopclosing.cpp
        core/monoinitializer.cpp
        core/system.cpp
        core/tracking.cpp
        )

add_library(vi_slam SHARED ${SOURCES})

target_link_libraries( vi_slam
        #vilib::vilib
        ${THIRD_PARTY_LIBS}
        )


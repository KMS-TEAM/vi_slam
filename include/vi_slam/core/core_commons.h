//
// Created by lacie on 25/05/2021.
//

#ifndef VI_SLAM_CORE_COMMONS_H
#define VI_SLAM_CORE_COMMONS_H

#include "vi_slam/common_include.h"
#include "vi_slam/datastructures/frame.h"
#include "vi_slam/datastructures/keyframe.h"

#include <g2o/types/sim3/types_seven_dof_expmap.h>

namespace vi_slam{

    namespace datastructures{
        class Frame;
    }
    namespace core{

        typedef std::map<datastructures::KeyFrame*,g2o::Sim3,std::less<datastructures::KeyFrame*>,
                Eigen::aligned_allocator<std::pair<datastructures::KeyFrame* const, g2o::Sim3> > > KeyFrameAndPose;

    }
}


#endif //VI_SLAM_CORE_COMMONS_H

//
// Created by lacie on 25/05/2021.
//

#ifndef VI_SLAM_VO_COMMONS_H
#define VI_SLAM_VO_COMMONS_H

#include "../common_include.h"
#include "frame.h"

namespace vi_slam{
    namespace core{

        cv::Mat getMotionFromFrame1to2(const Frame::Ptr f1, const Frame::Ptr f2);
        void getMotionFromFrame1to2(const Frame::Ptr f1, const Frame::Ptr f2, cv::Mat &R, cv::Mat &t);

    }
}


#endif //VI_SLAM_VO_COMMONS_H

//
// Created by lacie on 26/05/2021.
//

#ifndef VI_SLAM_FAST_CUDA_H
#define VI_SLAM_FAST_CUDA_H

#include "../common_include.h"
#include "vi_slam/basics/config.h"

#include "vilib/feature_detection/detector_base.h"
#include "vilib/feature_detection/detector_base_gpu.h"
#include "vilib/timer.h"
#include "vilib/statistics.h"


namespace vi_slam{
    namespace geometry{
        class FAST{
        public:
            FAST(){};
            void detect(const cv::Mat &image, vector<cv::KeyPoint> &keypoints);
            std::shared_ptr<vilib::DetectorBaseGPU> detector_gpu_;
            ~FAST(){};
        };
    }
}

#endif //VI_SLAM_FAST_CUDA_H


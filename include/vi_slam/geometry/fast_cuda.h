//
// Created by lacie on 26/05/2021.
//

#ifndef VI_SLAM_FAST_CUDA_H
#define VI_SLAM_FAST_CUDA_H

#include "../common_include.h"
//#include "vi_slam/basics/config.h"

//#include "vilib/feature_detection/detector_base.h"
//#include "vilib/feature_detection/detector_base_gpu.h"
//#include "vilib/timer.h"
//#include "vilib/statistics.h"
#include <iostream>

namespace vi_slam{
    namespace geometry{
        class FAST{
        public:
            //FAST(){};
            //void display_image(const cv::Mat & image, const char * image_title) const;
            //void load_image(const cv::Mat &image, bool display_image, bool display_info);
            //void detect(const cv::Mat &image, vector<cv::KeyPoint> &keypoints);
            ~FAST(){};

        private:
            //cv::Mat image_;
            //unsigned int image_width_;
            //unsigned int image_height_;
            //unsigned int image_channels_;
            //std::size_t image_size_;
            //std::shared_ptr<vilib::DetectorBaseGPU> detector_gpu_;
        };
    }
}

#endif //VI_SLAM_FAST_CUDA_H


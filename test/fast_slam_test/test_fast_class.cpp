//
// Created by lacie on 26/05/2021.
//

#include <iostream>

#include "vi_slam/geometry/fast_cuda.h"
#include "vilib/feature_detection/detector_base.h"
#include "vilib/feature_detection/detector_base_gpu.h"
#include "vilib/timer.h"
#include "vilib/statistics.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

int main(int argc, char** argv){
    // std::string image_path = argv[1];
    cv::Mat img = cv::imread("/home/lacie/Github/vi_slam/test/images/chessboard_798_798.png", cv::IMREAD_COLOR);

    if(img.empty())
    {
        std::cout << "Could not read the image: " << " LOL" << std::endl;
        return 1;
    }

    vi_slam::geometry::FAST proce;
    std::vector<cv::KeyPoint> keypoints;
    proce.detect(img, keypoints);

    return 0;
}
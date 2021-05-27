//
// Created by lacie on 26/05/2021.
//

#include <iostream>

#include "vi_slam/geometry/fast_cuda.h"
#include "vi_slam/basics/config.h"
#include "vi_slam/basics/yaml.h"

#include "vilib/feature_detection/detector_base.h"
#include "vilib/feature_detection/detector_base_gpu.h"
#include "vilib/timer.h"
#include "vilib/statistics.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>


int main(int argc, char** argv){
    // std::string image_path = argv[1];

    const string kConfigFile = "/home/lacie/Github/vi_slam/config/config.yaml";
    vi_slam::basics::Yaml config(kConfigFile);              // Use Yaml to read .yaml
    vi_slam::basics::Config::setParameterFile(kConfigFile); // Use Config to read .yaml

    cv::Mat img = cv::imread("/home/lacie/Github/vi_slam/test/images/chessboard_798_798.png", cv::IMREAD_COLOR);

    if(img.empty())
    {
        std::cout << "Could not read the image: " << " LOL" << std::endl;
        return 1;
    }
    cv::imshow("Image" , img);
    cv::waitKey(0);
    vi_slam::geometry::FAST proce;
    std::vector<cv::KeyPoint> keypoints;
    proce.detect(img, keypoints);

    return 0;
}
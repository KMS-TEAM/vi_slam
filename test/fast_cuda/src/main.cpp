#include <iostream>
#include <vector>
#include <memory>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "vilib/cuda_common.h"
#include "../include/common.h"
#include "../include/test_base.h"
#include "../include//arguments.h"

#include "../include/test_fast.h"

#define TEST_IMAGE_CHESSBOARD_798_798          "/home/lacie/Github/vi_slam/test/fast_cuda/images/chessboard_798_798.png"


using namespace vilib;

int main(){

    std::cout << "fast Cuda test" << std::endl;

    if(!cuda_initialize()) {
        return -1;
    }

    std::vector<struct TestCase> fast_test;

    fast_test.emplace_back(new TestFAST(TEST_IMAGE_CHESSBOARD_798_798,1));
    std::string image_title = "FAST test";

    cv::Mat origin_image;
    // harris_test.at(0).test_->load_image_to(TEST_IMAGE_CHESSBOARD_798_798, cv::IMREAD_COLOR, origin_image);
    // cv::imshow("test", origin_image);
    // cv::waitKey();

    // harris_test.at(0).test_->load_image(cv::IMREAD_COLOR, true, true);
    fast_test.at(0).test_->evaluate();

    cv::Mat result_image;
    fast_test.at(0).test_->load_image_to(TEST_IMAGE_CHESSBOARD_798_798, cv::IMREAD_COLOR, result_image);
    
    cv::imshow("test", result_image);
    cv::waitKey();
    

    return 0;
}
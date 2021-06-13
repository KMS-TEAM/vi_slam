//
// Created by lacie on 26/05/2021.
//

#include "vi_slam/geometry/fast_cuda.h"
#include "vi_slam/basics/config.h"
#include "vi_slam/geometry/fmatcher.h"

#include "vilib/preprocess/pyramid.h"
#include "vilib/storage/pyramid_pool.h"
#include "vilib/feature_detection/fast/rosten/fast_cpu.h"
#include "vilib/feature_detection/fast/fast_gpu.h"
#include "vilib/config.h"
#include "vilib/timer.h"
#include "vilib/statistics.h"

#include <cuda_runtime_api.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace vilib;

// Frame preprocessing
#define PYRAMID_LEVELS                       1
#define PYRAMID_MIN_LEVEL                    0
#define PYRAMID_MAX_LEVEL                    PYRAMID_LEVELS

// FAST detector parameters
#define FAST_EPSILON                         (10.0f)
#define FAST_MIN_ARC_LENGTH                  10
// Remark: the Rosten CPU version only works with
//         SUM_OF_ABS_DIFF_ON_ARC and MAX_THRESHOLD
#define FAST_SCORE                           SUM_OF_ABS_DIFF_ON_ARC

// NMS parameters
#define HORIZONTAL_BORDER                    0
#define VERTICAL_BORDER                      0
#define CELL_SIZE_WIDTH                      32
#define CELL_SIZE_HEIGHT                     32

namespace vi_slam{
    namespace geometry{

        void FAST::load_image(const cv::Mat &image, bool display_image, bool display_info) {

            cv::Mat temp = image.clone();
            cvtColor(temp,temp, cv::COLOR_BGR2GRAY);
            // cv::Mat temp_(temp.cols, temp.rows, CV_8UC(1));
            image_ = temp.clone();

//            for(int i = 0; i < temp.cols; i++){
//                for(int j = 0; j < temp.rows; j++){
//                    image_.at<unsigned char>(i, j) = temp.at<unsigned char>(i, j);
//                }
//            }
            if(display_image) {
                this->display_image(image_,"Original image");
            }
            image_width_ = image_.cols;
            image_height_ = image_.rows;
            image_channels_ = image_.channels();
            image_size_ = image_width_ * image_height_ * image_channels_;
            if(display_info) {
                std::cout << "Image width: " << image_width_ << " px" << std::endl;
                std::cout << "Image height: " << image_height_ << " px" << std::endl;
                std::cout << "Image channels: " << image_channels_ << std::endl;
                std::cout << "Image size: " << image_size_ << " bytes" << std::endl;
            }
        }
        void FAST::detect(const cv::Mat &image, vector<cv::KeyPoint> &keypoints) {

            // -- Set arguments

//            static const int pyramid_levels = basics::Config::get<int>("pyramid_levels");
//            static const int pyramid_min_level = basics::Config::get<int>("pyramid_min_level");
//            static const int pyramid_max_level = basics::Config::get<int>("pyramid_max_level");
//            static const double fast_epsilon = basics::Config::get<double>("fast_epsilon");
//            static const int fast_min_arc_length = basics::Config::get<int>("fast_min_arc_length");
//            static const int fast_score = basics::Config::get<int>("fast_score");
//            static const int horizontal_border = basics::Config::get<int>("horizontal_border");
//            static const int vertical_border = basics::Config::get<int>("vertical_border");
//            static const int cell_size_width = basics::Config::get<int>("cell_size_width");
//            static const int cell_size_height = basics::Config::get<int>("cell_size_height");

            load_image(image, true, true);

            // -- Create FAST detector
            detector_gpu_.reset(new FASTGPU(image_width_,
                                            image_height_,
                                            CELL_SIZE_WIDTH,
                                            CELL_SIZE_HEIGHT,
                                            PYRAMID_MIN_LEVEL,
                                            PYRAMID_MAX_LEVEL,
                                            HORIZONTAL_BORDER,
                                            VERTICAL_BORDER,
                                            FAST_EPSILON,
                                            FAST_MIN_ARC_LENGTH,
                                            FAST_SCORE));



            // Initialize the pyramid pool
            PyramidPool::init(1,
                              image_width_,
                              image_height_,
                              1,  // grayscale
                              PYRAMID_LEVELS,
                              IMAGE_PYRAMID_MEMORY_TYPE);

            // Create a Frame (image upload, pyramid)
            std::shared_ptr<Frame> frame0(new Frame(image_,0,PYRAMID_LEVELS));

            // Reset detector's grid
            // Note: this step could be actually avoided with custom processing
            detector_gpu_->reset();

            // Do the detection
            detector_gpu_->detect(frame0->pyramid_);
            std::cout << "Check" << std::endl;

            PyramidPool::deinit();

            auto & points_gpu = detector_gpu_->getPoints();
            auto & points_gpu_grid = detector_gpu_->getGrid();

            std::cout << "check keypoint size: " << points_gpu.size() << std::endl;
            for (std::size_t i = 0; i < points_gpu.size(); i++){
                std::cout << points_gpu[i].x_ << " " << points_gpu[i].y_ << " " << points_gpu[i].level_ << std::endl;
            }

            selectUniformKptsByGrid(keypoints, image_height_, image_width_);
        }

        void FAST::display_image(const cv::Mat & image, const char * image_title) const {
            cv::imshow(image_title, image);
            cv::waitKey();
        }
    }
}

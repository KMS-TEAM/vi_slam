//
// Created by lacie on 26/05/2021.
//

#include "vi_slam/geometry/fast_cuda.h"
#include "vi_slam/basics/config.h"
#include "vi_slam/geometry/feature_match.h"

#include "vilib/preprocess/pyramid.h"
#include "vilib/storage/pyramid_pool.h"
#include "vilib/feature_detection/fast/rosten/fast_cpu.h"
#include "vilib/feature_detection/fast/fast_gpu.h"
#include "vilib/config.h"
#include "vilib/timer.h"
#include "vilib/statistics.h"

#include <cuda_runtime_api.h>
#include <opencv2/imgproc.hpp>

using namespace vilib;

#define FAST_SCORE          SUM_OF_ABS_DIFF_ON_ARC

namespace vi_slam{
    namespace geometry{

        void FAST::detect(const cv::Mat &image, vector<cv::KeyPoint> &keypoints) {

            // -- Set arguments

            static const int pyramid_levels = basics::Config::get<int>("pyramid_levels");
            static const int pyramid_min_level = basics::Config::get<int>("pyramid_min_level");
            static const int pyramid_max_level = basics::Config::get<int>("pyramid_max_level");
            static const double fast_epsilon = basics::Config::get<double>("fast_epsilon");
            static const int fast_min_arc_length = basics::Config::get<int>("fast_min_arc_length");
            //static const int fast_score = basics::Config::get<int>("fast_score");
            static const int horizontal_border = basics::Config::get<int>("horizontal_border");
            static const int vertical_border = basics::Config::get<int>("vertical_border");
            static const int cell_size_width = basics::Config::get<int>("cell_size_width");
            static const int cell_size_height = basics::Config::get<int>("cell_size_height");


            // -- Create FAST detector
            detector_gpu_.reset(new FASTGPU(image.cols,
                                            image.rows,
                                            cell_size_width,
                                            cell_size_height,
                                            pyramid_min_level,
                                            pyramid_max_level,
                                            horizontal_border,
                                            vertical_border,
                                            fast_epsilon,
                                            fast_min_arc_length,
                                            FAST_SCORE));



            // Initialize the pyramid pool
            PyramidPool::init(1,
                              image.cols,
                              image.rows,
                              1,  // grayscale
                              pyramid_levels,
                              IMAGE_PYRAMID_MEMORY_TYPE);

            // Create a Frame (image upload, pyramid)
            std::shared_ptr<Frame> frame0(new Frame(image,0,pyramid_levels));

            // Reset detector's grid
            // Note: this step could be actually avoided with custom processing
            detector_gpu_->reset();

            // Do the detection
            detector_gpu_->detect(frame0->pyramid_);
            std::cout << "Check" << std::endl;

            PyramidPool::deinit();

            auto & points_gpu = detector_gpu_->getPoints();
            auto & points_gpu_grid = detector_gpu_->getGrid();

            std::cout << points_gpu.size() << std::endl;
//            for (std::size_t i = 0; i < points_gpu.size(); i++){
//                std::cout << points_gpu[i]
//            }

            selectUniformKptsByGrid(keypoints, image.rows, image.cols);
        }
    }
}

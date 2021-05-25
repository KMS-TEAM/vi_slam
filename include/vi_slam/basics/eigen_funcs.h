//
// Created by lacie on 25/05/2021.
//

// Eigen geometrical transformations and datatype conversions to/from OpenCV

#ifndef VI_SLAM_EIGEN_FUNCS_H
#define VI_SLAM_EIGEN_FUNCS_H

#include "../common_include.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <sophus/so3.hpp>
#include <sophus/se3.hpp>

#include <opencv2/core/eigen.hpp>

namespace vi_slam{
    namespace basics{
        /**
         * ------------------Eigen-----------------------
         * Get Affined3d using pos(x, y,z) and axis-rotation (rot_axis_x, rot_axis_y, rot_axis_z) with maginitude
         */
        Eigen::Affine3d getAffine3d(double x, double y, double z, double rot_axis_x, double rot_axis_y, double rot_axis_z);

        /**
         * -----------------OpenCV <--> Eigen-------------------
         * Convert cv::Mat R and t ->> Eigen Affined3d
         */
        Eigen::Affine3d transT_CVRt_to_EigenAffine3d(const cv::Mat &R, const cv::Mat &t);

        /**
         * ----------------OpenCV <--> Sophus-------------------
         */
        Sophus::SE3 transT_cv2sophus(const cv::Mat &T_cv);

        /**
         * ----------------OpenCV <--> Sophus-------------------
         */
        cv::Mat transT_sophus2cv(const Sophus::SE3 &T_sophus);
    }
}

#endif //VI_SLAM_EIGEN_FUNCS_H

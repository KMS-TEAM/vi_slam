//
// Created by lacie on 25/05/2021.
//

/** @brief Transformation matrices related to camera.
 */

#ifndef VI_SLAM_CAMERA_H
#define VI_SLAM_CAMERA_H

#include "../common_include.h"

namespace vi_slam{
    namespace geometry{

        //-----------------------transformations (by OpenCV)----------------
        cv::Point2f pixel2CamNormPlane(const cv::Point2f &p, const cv::Mat &K);
        cv::Point3f pixel2cam(const cv::Point2f &p, const cv::Mat &K, double depth = 1);
        cv::Point2f cam2pixel(const cv::Point3f &p, const cv::Mat &K);
        cv::Point2f cam2pixel(const cv::Mat &p, const cv::Mat &K);
        cv::Mat world2camera(const cv::Point3f &p, const cv::Mat &T_world_to_cam);

        //----------------------Camera Class-----------------------------
        class Camera{
        public:
            typedef std::shared_ptr<Camera> Ptr;
            double fx_, fy_, cx_, cy_;
            cv::Mat K_;
        public:
            Camera(double fx, double fy, double cx, double cy) : fx_(fx), fy_(fy), cx_(cx), cy_(cy){
                K_ = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
            }

            Camera(cv::Mat K){
                fx_ = K.at<double>(0, 0);
                fy_ = K.at<double>(1, 1);
                cx_ = K.at<double>(0, 2);
                cy_ = K.at<double>(1, 2);
                K_ = K;
            }
        };
    }
}

#endif //VI_SLAM_CAMERA_H

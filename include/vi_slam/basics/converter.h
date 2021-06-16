//
// Created by lacie on 05/06/2021.
//

#ifndef VI_SLAM_CONVERTER_H
#define VI_SLAM_CONVERTER_H

#include "../common_include.h"

#include <Eigen/Dense>
#include "g2o/g2o/types/slam3d/se3quat.h"
#include "g2o/g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/g2o/types/sim3/types_seven_dof_expmap.h"

namespace vi_slam{
    namespace basics{
        class converter{
        public:
            static std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors);

            static g2o::SE3Quat toSE3Quat(const cv::Mat &cvT){
                Eigen::Matrix<double,3,3> R;
                R << cvT.at<float>(0,0), cvT.at<float>(0,1), cvT.at<float>(0,2),
                        cvT.at<float>(1,0), cvT.at<float>(1,1), cvT.at<float>(1,2),
                        cvT.at<float>(2,0), cvT.at<float>(2,1), cvT.at<float>(2,2);

                Eigen::Matrix<double,3,1> t(cvT.at<float>(0,3), cvT.at<float>(1,3), cvT.at<float>(2,3));

                return g2o::SE3Quat(R,t);
            }
            static g2o::SE3Quat toSE3Quat(const g2o::Sim3 &gSim3);

            static cv::Mat toCvMat(const g2o::SE3Quat &SE3);
            static cv::Mat toCvMat(const g2o::Sim3 &Sim3);
            static cv::Mat toCvMat(const Eigen::Matrix<double,4,4> &m);
            static cv::Mat toCvMat(const Eigen::Matrix3d &m);
            static cv::Mat toCvMat(const Eigen::Matrix<double,3,1> &m);
            static cv::Mat toCvSE3(const Eigen::Matrix<double,3,3> &R, const Eigen::Matrix<double,3,1> &t);

            static Eigen::Matrix<double,3,1> toVector3d(const cv::Mat &cvVector);
            static Eigen::Matrix<double,3,1> toVector3d(const cv::Point3f &cvPoint);
            static Eigen::Matrix<double,3,3> toMatrix3d(const cv::Mat &cvMat3);

            static std::vector<float> toQuaternion(const cv::Mat &M);
        };

    }
}

#endif //VI_SLAM_CONVERTER_H

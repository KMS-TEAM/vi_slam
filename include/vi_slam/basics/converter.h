//
// Created by lacie on 05/06/2021.
//

#ifndef VI_SLAM_CONVERTER_H
#define VI_SLAM_CONVERTER_H

#include "vi_slam/common_include.h"

#include <Eigen/Dense>
#include <g2o/types/slam3d/se3quat.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>

namespace vi_slam{
    namespace basics{
        class converter{
            public:
            static std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors);

            static g2o::SE3Quat toSE3Quat(const cv::Mat &cvT);
            static g2o::SE3Quat toSE3Quat(const g2o::Sim3 &gSim3);

            static cv::Mat toCvMat(const g2o::SE3Quat &SE3);
            static cv::Mat toCvMat(const g2o::Sim3 &Sim3);
            static cv::Mat toCvMat(const Eigen::Matrix<double,4,4> &m);
            static cv::Mat toCvMat(const Eigen::Matrix3d &m);
            static cv::Mat toCvMat(const Eigen::Matrix<double,3,1> &m);
            static cv::Mat toCvMat(const Eigen::MatrixXd &m);
            static cv::Mat toCvSE3(const Eigen::Matrix<double,3,3> &R, const Eigen::Matrix<double,3,1> &t);
            static cv::Mat tocvSkewMatrix(const cv::Mat &v);
            static Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& w);


            static Eigen::Matrix<double,3,1> toVector3d(const cv::Mat &cvVector);
            static Eigen::Matrix<double,3,1> toVector3d(const cv::Point3f &cvPoint);
            static Eigen::Matrix<double,3,3> toMatrix3d(const cv::Mat &cvMat3);
            static Eigen::Matrix<double,4,4> toMatrix4d(const cv::Mat &cvMat4);
            static std::vector<float> toQuaternion(const cv::Mat &M);

            static void quaternionNormalize(Eigen::Vector4d& q);
            static Eigen::Vector4d quaternionMultiplication (const Eigen::Vector4d& q1,
                                                             const Eigen::Vector4d& q2);
            static Eigen::Vector4d smallAngleQuaternion(const Eigen::Vector3d& dtheta);
            static Eigen::Matrix3d quaternionToRotation(const Eigen::Vector4d& q);
            static Eigen::Vector4d rotationToQuaternion(const Eigen::Matrix3d& R);


            static bool isRotationMatrix(const cv::Mat &R);
            static std::vector<float> toEuler(const cv::Mat &R);
        };
        
    }
}

#endif //VI_SLAM_CONVERTER_H

//
// Created by lacie on 25/05/2021.
//

/** @brief Transformation matrices related to camera.
 */

#ifndef VI_SLAM_CAMERA_H
#define VI_SLAM_CAMERA_H

#include "vi_slam/common_include.h"

#include <opencv2/core/core.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/assume_abstract.hpp>

#include <Eigen/Geometry>

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
            Camera() {}
            Camera(const std::vector<float> &_vParameters) : mvParameters(_vParameters) {}
            ~Camera() {}

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

            virtual cv::Point2f project(const cv::Point3f &p3D) = 0;
            virtual cv::Point2f project(const cv::Matx31f &m3D) = 0;
            virtual cv::Point2f project(const cv::Mat& m3D) = 0;
            virtual Eigen::Vector2d project(const Eigen::Vector3d & v3D) = 0;
            virtual cv::Mat projectMat(const cv::Point3f& p3D) = 0;

            virtual float uncertainty2(const Eigen::Matrix<double,2,1> &p2D) = 0;

            virtual cv::Point3f unproject(const cv::Point2f &p2D) = 0;
            virtual cv::Mat unprojectMat(const cv::Point2f &p2D) = 0;
            virtual cv::Matx31f unprojectMat_(const cv::Point2f &p2D) = 0;

            virtual cv::Mat projectJac(const cv::Point3f &p3D) = 0;
            virtual Eigen::Matrix<double,2,3> projectJac(const Eigen::Vector3d& v3D) = 0;

            virtual cv::Mat unprojectJac(const cv::Point2f &p2D) = 0;

            virtual bool ReconstructWithTwoViews(const std::vector<cv::KeyPoint>& vKeys1, const std::vector<cv::KeyPoint>& vKeys2, const std::vector<int> &vMatches12,
                                                 cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated) = 0;

            virtual cv::Mat toK() = 0;
            virtual cv::Matx33f toK_() = 0;

            virtual bool epipolarConstrain(Camera* otherCamera, const cv::KeyPoint& kp1, const cv::KeyPoint& kp2, const cv::Mat& R12, const cv::Mat& t12, const float sigmaLevel, const float unc) = 0;
            virtual bool epipolarConstrain_(Camera* otherCamera, const cv::KeyPoint& kp1, const cv::KeyPoint& kp2, const cv::Matx33f& R12, const cv::Matx31f& t12, const float sigmaLevel, const float unc) = 0;

            float getParameter(const int i){return mvParameters[i];}
            void setParameter(const float p, const size_t i){mvParameters[i] = p;}

            size_t size(){return mvParameters.size();}

            virtual bool matchAndtriangulate(const cv::KeyPoint& kp1, const cv::KeyPoint& kp2, Camera* pOther,
                                             cv::Mat& Tcw1, cv::Mat& Tcw2,
                                             const float sigmaLevel1, const float sigmaLevel2,
                                             cv::Mat& x3Dtriangulated) = 0;

            unsigned int GetId() { return mnId; }

            unsigned int GetType() { return mnType; }

            const unsigned int CAM_PINHOLE = 0;
            const unsigned int CAM_FISHEYE = 1;

            static long unsigned int nNextId;

        public:
            double fx_, fy_, cx_, cy_;
            cv::Mat K_;

        protected:
            std::vector<float> mvParameters;

            unsigned int mnId;

            unsigned int mnType;
        };
    }
}

#endif //VI_SLAM_CAMERA_H

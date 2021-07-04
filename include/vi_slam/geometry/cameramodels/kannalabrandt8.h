//
// Created by lacie on 30/06/2021.
//

#ifndef VI_SLAM_KANNALABRANDT8_H
#define VI_SLAM_KANNALABRANDT8_H

#include <assert.h>
#include <vector>
#include <opencv2/core/core.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/assume_abstract.hpp>

#include "vi_slam/common_include.h"
#include "vi_slam/geometry/cameramodels/camera.h"

#include "vi_slam/geometry/motion_estimation.h"

namespace vi_slam{
    namespace geometry{

        class MotionEstimator;

        class KannalaBrandt8 : public Camera{
        public:
            KannalaBrandt8() : precision(1e-6) {
                mvParameters.resize(8);
                mnId=nNextId++;
                mnType = CAM_FISHEYE;
            }
            KannalaBrandt8(const std::vector<float> _vParameters) : Camera(_vParameters), precision(1e-6), mvLappingArea(2,0) ,tvr(nullptr) {
                assert(mvParameters.size() == 8);
                mnId=nNextId++;
                mnType = CAM_FISHEYE;
            }

            KannalaBrandt8(const std::vector<float> _vParameters, const float _precision) : Camera(_vParameters),
                                                                                            precision(_precision), mvLappingArea(2,0) {
                assert(mvParameters.size() == 8);
                mnId=nNextId++;
                mnType = CAM_FISHEYE;
            }
            KannalaBrandt8(KannalaBrandt8* pKannala) : Camera(pKannala->mvParameters), precision(pKannala->precision), mvLappingArea(2,0) ,tvr(nullptr) {
                assert(mvParameters.size() == 8);
                mnId=nNextId++;
                mnType = CAM_FISHEYE;
            }

            cv::Point2f project(const cv::Point3f &p3D);
            cv::Point2f project(const cv::Matx31f &m3D);
            cv::Point2f project(const cv::Mat& m3D);
            Eigen::Vector2d project(const Eigen::Vector3d & v3D);
            cv::Mat projectMat(const cv::Point3f& p3D);

            float uncertainty2(const Eigen::Matrix<double,2,1> &p2D);

            cv::Point3f unproject(const cv::Point2f &p2D);
            cv::Mat unprojectMat(const cv::Point2f &p2D);
            cv::Matx31f unprojectMat_(const cv::Point2f &p2D);

            cv::Mat projectJac(const cv::Point3f &p3D);
            Eigen::Matrix<double,2,3> projectJac(const Eigen::Vector3d& v3D);

            cv::Mat unprojectJac(const cv::Point2f &p2D);

            bool ReconstructWithTwoViews(const std::vector<cv::KeyPoint>& vKeys1, const std::vector<cv::KeyPoint>& vKeys2, const std::vector<int> &vMatches12,
                                         cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated);

            cv::Mat toK();
            cv::Matx33f toK_();

            bool epipolarConstrain(Camera* pCamera2, const cv::KeyPoint& kp1, const cv::KeyPoint& kp2, const cv::Mat& R12, const cv::Mat& t12, const float sigmaLevel, const float unc);
            bool epipolarConstrain_(Camera* pCamera2, const cv::KeyPoint& kp1, const cv::KeyPoint& kp2, const cv::Matx33f& R12, const cv::Matx31f& t12, const float sigmaLevel, const float unc);


            float TriangulateMatches(Camera* pCamera2, const cv::KeyPoint& kp1, const cv::KeyPoint& kp2, const cv::Mat& R12, const cv::Mat& t12, const float sigmaLevel, const float unc, cv::Mat& p3D);
            float TriangulateMatches_(Camera* pCamera2, const cv::KeyPoint& kp1, const cv::KeyPoint& kp2, const cv::Matx33f& R12, const cv::Matx31f& t12, const float sigmaLevel, const float unc, cv::Matx31f& p3D);

            std::vector<int> mvLappingArea;

            bool matchAndtriangulate(const cv::KeyPoint& kp1, const cv::KeyPoint& kp2, Camera* pOther,
                                     cv::Mat& Tcw1, cv::Mat& Tcw2,
                                     const float sigmaLevel1, const float sigmaLevel2,
                                     cv::Mat& x3Dtriangulated);

            friend std::ostream& operator<<(std::ostream& os, const KannalaBrandt8& kb);
            friend std::istream& operator>>(std::istream& is, KannalaBrandt8& kb);
        private:
            const float precision;

            //Parameters vector corresponds to
            //[fx, fy, cx, cy, k0, k1, k2, k3]

            MotionEstimator* tvr;

            void Triangulate(const cv::Point2f &p1, const cv::Point2f &p2, const cv::Mat &Tcw1, const cv::Mat &Tcw2,cv::Mat &x3D);
            void Triangulate_(const cv::Point2f &p1, const cv::Point2f &p2, const cv::Matx44f &Tcw1, const cv::Matx44f &Tcw2,cv::Matx31f &x3D);
        };
    }
}

#endif //VI_SLAM_KANNALABRANDT8_H

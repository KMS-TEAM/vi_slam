//
// Created by lacie on 30/06/2021.
//

#ifndef VI_SLAM_PINHOLE_H
#define VI_SLAM_PINHOLE_H

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

        class Pinhole : public Camera {
            public:
                Pinhole() {
                    mvParameters.resize(4);
                    mnId=nNextId++;
                    mnType = CAM_PINHOLE;
                }
                Pinhole(const std::vector<float> _vParameters) : Camera(_vParameters), tvr(nullptr) {
                    assert(mvParameters.size() == 4);
                    mnId=nNextId++;
                    mnType = CAM_PINHOLE;
                }

                Pinhole(Pinhole* pPinhole) : Camera(pPinhole->mvParameters), tvr(nullptr) {
                    assert(mvParameters.size() == 4);
                    mnId=nNextId++;
                    mnType = CAM_PINHOLE;
                }


                ~Pinhole(){
                    if(tvr) delete tvr;
                }

                cv::Point2f project(const cv::Point3f &p3D);
                cv::Point2f project(const cv::Matx31f &m3D);
                cv::Point2f project(const cv::Mat &m3D);
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

                bool matchAndtriangulate(const cv::KeyPoint& kp1, const cv::KeyPoint& kp2, Camera* pOther,
                                         cv::Mat& Tcw1, cv::Mat& Tcw2,
                                         const float sigmaLevel1, const float sigmaLevel2,
                                         cv::Mat& x3Dtriangulated) { return false;}

                friend std::ostream& operator<<(std::ostream& os, const Pinhole& ph);
                friend std::istream& operator>>(std::istream& os, Pinhole& ph);
            private:
                cv::Mat SkewSymmetricMatrix(const cv::Mat &v);
                cv::Matx33f SkewSymmetricMatrix_(const cv::Matx31f &v);

                //Parameters vector corresponds to
                //      [fx, fy, cx, cy]

                MotionEstimator* tvr;
        };
    }
}

#endif //VI_SLAM_PINHOLE_H

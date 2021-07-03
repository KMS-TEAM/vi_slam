//
// Created by lacie on 12/06/2021.
//

#ifndef VI_SLAM_SIM3SOLVER_H
#define VI_SLAM_SIM3SOLVER_H

#include "vi_slam/common_include.h"
#include "vi_slam/datastructures/keyframe.h"
#include "vi_slam/datastructures/mappoint.h"
#include "vi_slam/geometry/cameramodels/camera.h"

namespace vi_slam{

    namespace datastructures{
        class KeyFrame;
        class MapPoint;
    }
    namespace geometry{
        class Camera;
    }

    namespace optimization{

        using namespace datastructures;
        using namespace geometry;
        
        class Sim3Solver
        {
        public:

            Sim3Solver(KeyFrame* pKF1, KeyFrame* pKF2, const std::vector<MapPoint*> &vpMatched12, const bool bFixScale = true,
                       const vector<KeyFrame*> vpKeyFrameMatchedMP = vector<KeyFrame*>());

            void SetRansacParameters(double probability = 0.99, int minInliers = 6 , int maxIterations = 300);

            cv::Mat find(std::vector<bool> &vbInliers12, int &nInliers);

            cv::Mat iterate(int nIterations, bool &bNoMore, std::vector<bool> &vbInliers, int &nInliers);
            cv::Mat iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers, bool &bConverge);

            cv::Mat GetEstimatedRotation();
            cv::Mat GetEstimatedTranslation();
            float GetEstimatedScale();


        protected:

            void ComputeCentroid(cv::Mat &P, cv::Mat &Pr, cv::Mat &C);

            void ComputeSim3(cv::Mat &P1, cv::Mat &P2);

            void CheckInliers();

            void Project(const std::vector<cv::Mat> &vP3Dw, std::vector<cv::Mat> &vP2D, cv::Mat Tcw, Camera* pCamera);
            void FromCameraToImage(const std::vector<cv::Mat> &vP3Dc, std::vector<cv::Mat> &vP2D, Camera* pCamera);


        protected:

            // KeyFrames and matches
            KeyFrame* mpKF1;
            KeyFrame* mpKF2;

            std::vector<cv::Mat> mvX3Dc1;
            std::vector<cv::Mat> mvX3Dc2;
            std::vector<MapPoint*> mvpMapPoints1;
            std::vector<MapPoint*> mvpMapPoints2;
            std::vector<MapPoint*> mvpMatches12;
            std::vector<size_t> mvnIndices1;
            std::vector<size_t> mvSigmaSquare1;
            std::vector<size_t> mvSigmaSquare2;
            std::vector<size_t> mvnMaxError1;
            std::vector<size_t> mvnMaxError2;

            int N;
            int mN1;

            // Current Estimation
            cv::Mat mR12i;
            cv::Mat mt12i;
            float ms12i;
            cv::Mat mT12i;
            cv::Mat mT21i;
            std::vector<bool> mvbInliersi;
            int mnInliersi;

            // Current Ransac State
            int mnIterations;
            std::vector<bool> mvbBestInliers;
            int mnBestInliers;
            cv::Mat mBestT12;
            cv::Mat mBestRotation;
            cv::Mat mBestTranslation;
            float mBestScale;

            // Scale is fixed to 1 in the stereo/RGBD case
            bool mbFixScale;

            // Indices for random selection
            std::vector<size_t> mvAllIndices;

            // Projections
            std::vector<cv::Mat> mvP1im1;
            std::vector<cv::Mat> mvP2im2;

            // RANSAC probability
            double mRansacProb;

            // RANSAC min inliers
            int mRansacMinInliers;

            // RANSAC max iterations
            int mRansacMaxIts;

            // Threshold inlier/outlier. e = dist(Pi,T_ij*Pj)^2 < 5.991*mSigma2
            float mTh;
            float mSigma2;

            // Calibration
            cv::Mat mK1;
            cv::Mat mK2;

            Camera* pCamera1, *pCamera2;
        };
    }
}
#endif //VI_SLAM_SIM3SOLVER_H

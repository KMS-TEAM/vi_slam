//
// Created by lacie on 11/06/2021.
//

#ifndef VI_SLAM_MONOINITIALER_H
#define VI_SLAM_MONOINITIALER_H

#include "vi_slam/datastructures/frame.h"
#include "vi_slam/geometry/motion_estimation.h"
#include <opencv2/opencv.hpp>

using namespace std;
// using namespace vi_slam::datastructures;

namespace vi_slam{
    namespace datastructures{
        class Frame;
    }
    namespace geometry{
        class MotionEstimator;
    }
    namespace core{
        class MonoInitializer {
            typedef pair<int,int> Match;

        public:
            // Fix the reference frame
            MonoInitializer(const datastructures::Frame &ReferenceFrame, float sigma = 1.0, int iterations = 200);

            // Computes in parallel a fundamental matrix and a homography
            // Selects a model and tries to recover the motion and the structure from motion
            bool Initialize(const datastructures::Frame &CurrentFrame, const vector<int> &vMatches12,
                            cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated);

            ~MonoInitializer(){};

        private:

            void FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21);
            void FindFundamental(vector<bool> &vbInliers, float &score, cv::Mat &F21);

            cv::Mat ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2);
            cv::Mat ComputeF21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2);

            float CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, float sigma);

            float CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma);

            bool ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                              cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated);

            bool ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                              cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated);

            void Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);

            void Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);
            // void Normalize(const vector<cv::Point2f> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);

            int CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                        const vector<Match> &vMatches12, vector<bool> &vbInliers,
                        const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax);

            void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);

        private:
            // Keypoints from Reference Frame (Frame 1)
            vector<cv::KeyPoint> mvKeys1;

            // Keypoints from Current Frame (Frame 2)
            vector<cv::KeyPoint> mvKeys2;

            // Current Matches from Reference to Current
            vector<Match> mvMatches12;
            vector<bool> mvbMatched1;

            // Calibration
            cv::Mat mK;

            // Standard Deviation and Variance
            float mSigma, mSigma2;

            // Ransac max iterations
            int mMaxIterations;

            // Ransac sets
            vector<vector<size_t> > mvSets;

            geometry::Camera* mpCamera;
        };
    }
}

#endif //VI_SLAM_MONOINITIALER_H

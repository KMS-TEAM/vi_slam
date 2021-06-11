//
// Created by lacie on 11/06/2021.
//

#ifndef VI_SLAM_MONOINITIALER_H
#define VI_SLAM_MONOINITIALER_H

#include "vi_slam/datastructures/frame.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace vi_slam::datastructures;

namespace vi_slam{
    namespace core{
        class MonoInitializer {
            typedef pair<int,int> Match;

        public:
            // Fix the reference frame
            MonoInitializer(const Frame &ReferenceFrame, float sigma = 1.0, int iterations = 200);

            // Computes in parallel a fundamental matrix and a homography
            // Selects a model and tries to recover the motion and the structure from motion
            bool Initialize(const Frame &CurrentFrame, const vector<int> &vMatches12,
                            cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated);

            ~MonoInitializer(){};

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
        };
    }
}

#endif //VI_SLAM_MONOINITIALER_H

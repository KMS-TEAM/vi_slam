//
// Created by lacie on 25/05/2021.
//

#ifndef VI_SLAM_FEATURE_MATCH_H
#define VI_SLAM_FEATURE_MATCH_H

#include "../common_include.h"

#include "vi_slam/datastructures/mappoint.h"
#include "vi_slam/datastructures/keyframe.h"
#include "vi_slam/datastructures/frame.h"

using namespace vi_slam::datastructures;
using namespace std;

namespace vi_slam{
    namespace geometry{

        void calcKeyPoints(const cv::Mat &image,
                           vector<cv::KeyPoint> &keypoints);

        /**
         * @brief Compute the descriptors of keypoints.
         * Meanwhile, keypoints might be changed.
         */
        void calcDescriptors(const cv::Mat &image,
                             vector<cv::KeyPoint> &keypoints,
                             cv::Mat &descriptors);

        void matchFeatures(
                const cv::Mat1b &descriptors_1, const cv::Mat1b &descriptors_2,
                vector<cv::DMatch> &matches,
                int method_index = 1,
                bool is_print_res = false,
                // Below are optional arguments for feature_matching_method_index==3
                const vector<cv::KeyPoint> &keypoints_1 = vector<cv::KeyPoint>(),
                const vector<cv::KeyPoint> &keypoints_2 = vector<cv::KeyPoint>(),
                float max_matching_pixel_dist = 0.0);

        vector<cv::DMatch> matchByRadiusAndBruteForce(
                const vector<cv::KeyPoint> &keypoints_1,
                const vector<cv::KeyPoint> &keypoints_2,
                const cv::Mat1b &descriptors_1,
                const cv::Mat1b &descriptors_2,
                float max_matching_pixel_dist);

        // Remove duplicate matches.
        // After cv's match func, many kpts in I1 might matched to a same kpt in I2.
        // Sorting the trainIdx(I2), and make the match unique.
        void removeDuplicatedMatches(vector<cv::DMatch> &matches);

        // Use a grid to remove the keypoints that are too close to each other.
        void selectUniformKptsByGrid(vector<cv::KeyPoint> &keypoints,
                                     int image_rows, int image_cols);

        // --------------------- Other assistant functions ---------------------
        double computeMeanDistBetweenKeypoints(
                const vector<cv::KeyPoint> &kpts1, const vector<cv::KeyPoint> &kpts2, const vector<cv::DMatch> &matches);

        // --------------------- Datatype conversion ---------------------
        vector<cv::DMatch> inliers2DMatches(const vector<int> inliers);
        vector<cv::KeyPoint> pts2Keypts(const vector<cv::Point2f> pts);

        class FMatcher{
        public:

            FMatcher(float nnratio=0.6, bool checkOri=true);

            // Computes the Hamming distance between two ORB descriptors
            static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

            // Search matches between Frame keypoints and projected MapPoints. Returns number of matches
            // Used to track the local map (Tracking)
            int SearchByProjection(Frame &F, const std::vector<MapPoint*> &vpMapPoints, const float th=3);

            // Project MapPoints tracked in last frame into the current frame and search matches.
            // Used to track from previous frame (Tracking)
            int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono);

            // Project MapPoints seen in KeyFrame into the Frame and search matches.
            // Used in relocalisation (Tracking)
            int SearchByProjection(Frame &CurrentFrame, KeyFrame* pKF, const std::set<MapPoint*> &sAlreadyFound, const float th, const int ORBdist);

            // Project MapPoints using a Similarity Transformation and search matches.
            // Used in loop detection (Loop Closing)
            int SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapPoint*> &vpPoints, std::vector<MapPoint*> &vpMatched, int th);

            // Search matches between MapPoints in a KeyFrame and ORB in a Frame.
            // Brute force constrained to ORB that belong to the same vocabulary node (at a certain level)
            // Used in Relocalisation and Loop Detection
            int SearchByBoW(KeyFrame *pKF, Frame &F, std::vector<MapPoint*> &vpMapPointMatches);
            int SearchByBoW(KeyFrame *pKF1, KeyFrame* pKF2, std::vector<MapPoint*> &vpMatches12);

            // Matching for the Map Initialization (only used in the monocular case)
            int SearchForInitialization(Frame &F1, Frame &F2, std::vector<cv::Point2f> &vbPrevMatched, std::vector<int> &vnMatches12, int windowSize=10);

            // Matching to triangulate new MapPoints. Check Epipolar Constraint.
            int SearchForTriangulation(KeyFrame *pKF1, KeyFrame* pKF2, cv::Mat F12,
                                       std::vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo);

            // Search matches between MapPoints seen in KF1 and KF2 transforming by a Sim3 [s12*R12|t12]
            // In the stereo and RGB-D case, s12=1
            int SearchBySim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint *> &vpMatches12, const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th);

            // Project MapPoints into KeyFrame and search for duplicated MapPoints.
            int Fuse(KeyFrame* pKF, const vector<MapPoint *> &vpMapPoints, const float th=3.0);

            // Project MapPoints into KeyFrame using a given Sim3 and search for duplicated MapPoints.
            int Fuse(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapPoint*> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint);

        public:

            static const int TH_LOW;
            static const int TH_HIGH;
            static const int HISTO_LENGTH;


        protected:

            bool CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12, const datastructures::KeyFrame *pKF);

            float RadiusByViewingCos(const float &viewCos);

            void ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);

            float mfNNratio;
            bool mbCheckOrientation;
        };

    }
}

#endif //VI_SLAM_FEATURE_MATCH_H

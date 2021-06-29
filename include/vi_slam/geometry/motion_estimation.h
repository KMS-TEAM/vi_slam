//
// Created by lacie on 25/05/2021.
//

#ifndef VI_SLAM_MOTION_ESTIMATION_H
#define VI_SLAM_MOTION_ESTIMATION_H

/** @brief Functions for estimating camera motion between two frames.
 */

#include "../common_include.h"
#include "vi_slam/geometry/fmatcher.h"
#include "vi_slam/geometry/epipolar_geometry.h"
#include "vi_slam/basics/opencv_funcs.h"
#include "vi_slam/geometry/cameramodels/camera.h"

namespace vi_slam{
    namespace geometry{

        class MotionEstimator{

        public:
            // Fix the reference frame
            MotionEstimator(cv::Mat& k, float sigma = 1.0, int iterations = 200);

            // Computes in parallel a fundamental matrix and a homography
            // Selects a model and tries to recover the motion and the structure from motion
            bool Reconstruct(const std::vector<cv::KeyPoint>& vKeys1, const std::vector<cv::KeyPoint>& vKeys2, const std::vector<int> &vMatches12,
                             cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated);

            ~MotionEstimator();

        private:
            typedef pair<int,int> Match;

            /** @brief This is a giant function, which:
             *      - Computes: E21/H21
             *      - Decompose E and H into R and t.
             *      - Do triangulation based on R and t.
             * @return list_R, list_t
             * @return list_matches
             * @return list_normal
             * @return sols_pts3d_in_cam1
             * @return int: the index of the best R and t in list_R and list_t.
             *      Whether choose E or H is based on the score defined in ORB-SLAM2 paper.
             *      If choose H, then choose the pose which has largest norm z in camera direction.
             */
            int helperEstimatePossibleRelativePosesByEpipolarGeometry(
                    const vector<cv::KeyPoint> &keypoints_1,
                    const vector<cv::KeyPoint> &keypoints_2,
                    const vector<cv::DMatch> &matches,
                    const cv::Mat &K, // camera intrinsics
                    vector<cv::Mat> &list_R, vector<cv::Mat> &list_t,
                    vector<vector<cv::DMatch>> &list_matches,
                    vector<cv::Mat> &list_normal,
                    vector<vector<cv::Point3f>> &sols_pts3d_in_cam1,
                    bool is_print_res = false,
                    bool is_calc_homo = true,
                    bool is_frame_cam2_to_cam1 = true);

            /** @brief Compute:
             *      - The error of eppipolar constraint
             *      - The error of triangulation
             */
            void helperEvalEppiAndTriangErrors(
                    const vector<cv::KeyPoint> &keypoints_1,
                    const vector<cv::KeyPoint> &keypoints_2,
                    const vector<vector<cv::DMatch>> &list_matches,
                    const vector<vector<cv::Point3f>> &sols_pts3d_in_cam1_by_triang,
                    const vector<cv::Mat> &list_R, const vector<cv::Mat> &list_t, const vector<cv::Mat> &list_normal,
                    const cv::Mat &K, // camera intrinsics
                    bool is_print_res);

            /** @brief Estimate camera motion by Essential matrix.
             * @return R: R_cam2_to_cam1
             * @return t: t_cam2_to_cam1
             * @return inlier_matches: list of indices of keypoints
             */
            void helperEstiMotionByEssential(
                    const vector<cv::KeyPoint> &keypoints_1,
                    const vector<cv::KeyPoint> &keypoints_2,
                    const vector<cv::DMatch> &matches,
                    const cv::Mat &K, // camera intrinsics
                    cv::Mat &R, cv::Mat &t,
                    vector<cv::DMatch> &inlier_matches,
                    bool is_print_res = false);

            /** @brief After feature matching, find inlier matches
             *      by using epipolar constraint to exclude wrong matches.
             * @param: keypoints_1, keypoints_2, matches, K
             * @return vector<cv::DMatch>: the inlier matches
             */
            vector<cv::DMatch> helperFindInlierMatchesByEpipolarCons(
                    const vector<cv::KeyPoint> &keypoints_1,
                    const vector<cv::KeyPoint> &keypoints_2,
                    const vector<cv::DMatch> &matches,
                    const cv::Mat &K // camera intrinsics
            );

            /** @brief Triangulate points.
             * @param: prev_kpts
             * @param: curr_kpts
             * @param: curr_inlier_matches (prev is queryIdx, curr is trainIdx)
             * @return vector<cv::Point3f>: triangulated points in current frame.
             */
            vector<cv::Point3f> helperTriangulatePoints(
                    const vector<cv::KeyPoint> &prev_kpts, const vector<cv::KeyPoint> &curr_kpts,
                    const vector<cv::DMatch> &curr_inlier_matches,
                    const cv::Mat &T_curr_to_prev,
                    const cv::Mat &K);

            vector<cv::Point3f> helperTriangulatePoints(
                    const vector<cv::KeyPoint> &prev_kpts, const vector<cv::KeyPoint> &curr_kpts,
                    const vector<cv::DMatch> &curr_inlier_matches,
                    const cv::Mat &R_curr_to_prev, const cv::Mat &t_curr_to_prev,
                    const cv::Mat &K);

            /**
             * @brief Compute the score of estiamted Essential matrix by the method in ORB-SLAM
            */
            double checkEssentialScore(const cv::Mat &E21, const cv::Mat &K,
                                       const vector<cv::Point2f> &pts_img1, const vector<cv::Point2f> &pts_img2,
                                       vector<int> &inliers_index, double sigma = 1.0);

            /**
             * @brief Compute the score of estiamted Homography matrix by the method in ORB-SLAM
             */
            double checkHomographyScore(const cv::Mat &H21,
                                        const vector<cv::Point2f> &pts_img1, const vector<cv::Point2f> &pts_img2,
                                        vector<int> &inliers_index, double sigma = 1.0);

            void FindHomography(vector<cv::KeyPoint> &mvKeys1,
                                vector<cv::KeyPoint> &mvKeys2,
                                vector<bool> &vbMatchesInliers,
                                vector<Match> &mvMatches12,
                                float &score, cv::Mat &H21,
                                int mMaxIterations, float mSigma,
                                vector<vector<size_t> > mvSets);

            void FindFundamental(vector<cv::KeyPoint> &mvKeys1,
                                 vector<cv::KeyPoint> &mvKeys2,
                                 vector<bool> &vbMatchesInliers,
                                 vector<Match> &mvMatches12,
                                 int mMaxIterations, float mSigma,
                                 float &score, cv::Mat &F21,
                                 vector<vector<size_t> > mvSets);

            cv::Mat ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2);

            cv::Mat ComputeF21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2);

            float CheckHomography(const cv::Mat &H21, const cv::Mat &H12,
                                  vector<bool> &vbMatchesInliers,
                                  vector<cv::KeyPoint> &mvKeys1,
                                  vector<cv::KeyPoint> &mvKeys2,
                                  vector<Match> &mvMatches12,
                                  float sigma);

            float CheckFundamental(const cv::Mat &F21,
                                   vector<bool> &vbMatchesInliers,
                                   vector<Match> &mvMatches12,
                                   vector<cv::KeyPoint> &mvKeys1,
                                   vector<cv::KeyPoint> &mvKeys2,
                                   float sigma);

            bool ReconstructF(vector<bool> &vbMatchesInliers,
                              vector<Match> &mvMatches12,
                              vector<cv::KeyPoint> &mvKeys1,
                              vector<cv::KeyPoint> &mvKeys2,
                              cv::Mat &F21, cv::Mat &K,
                              cv::Mat &R21, cv::Mat &t21,
                              vector<cv::Point3f> &vP3D,
                              vector<bool> &vbTriangulated,
                              float minParallax, float mSigma,
                              int minTriangulated);

            bool ReconstructH(vector<bool> &vbMatchesInliers,
                              vector<Match> &mvMatches12,
                              vector<cv::KeyPoint> &mvKeys1,
                              vector<cv::KeyPoint> &mvKeys2,
                              cv::Mat &H21, cv::Mat &K,
                              cv::Mat &R21, cv::Mat &t21,
                              vector<cv::Point3f> &vP3D,
                              vector<bool> &vbTriangulated,
                              float minParallax, float mSigma,
                              int minTriangulated);

            void Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);

            void Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);

            int CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                        const vector<Match> &vMatches12, vector<bool> &vbInliers,
                        const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax);

            void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);

            // ---------------------------------------------
            // ---------------------------------------------
            // ----------- debug functions -----------------
            // ---------------------------------------------
            // ---------------------------------------------

            void printResult_estiMotionByEssential(
                    const cv::Mat &essential_matrix,
                    const vector<int> &inliers_index,
                    const cv::Mat &R,
                    const cv::Mat &t);

            void printResult_estiMotionByHomography(
                    const cv::Mat &homography_matrix,
                    const vector<int> &inliers_index,
                    const vector<cv::Mat> &Rs, const vector<cv::Mat> &ts,
                    vector<cv::Mat> &normals);

            void print_EpipolarError_and_TriangulationResult_By_Common_Inlier(
                    const vector<cv::Point2f> &pts_img1, const vector<cv::Point2f> &pts_img2,
                    const vector<cv::Point2f> &pts_on_np1, const vector<cv::Point2f> &pts_on_np2,
                    const vector<vector<cv::Point3f>> &sols_pts3d_in_cam1,
                    const vector<vector<int>> &list_inliers,
                    const vector<cv::Mat> &list_R, const vector<cv::Mat> &list_t, const cv::Mat &K);

            void print_EpipolarError_and_TriangulationResult_By_Solution(
                    const vector<cv::Point2f> &pts_img1, const vector<cv::Point2f> &pts_img2,
                    const vector<cv::Point2f> &pts_on_np1, const vector<cv::Point2f> &pts_on_np2,
                    const vector<vector<cv::Point3f>> &sols_pts3d_in_cam1,
                    const vector<vector<int>> &list_inliers,
                    const vector<cv::Mat> &list_R, const vector<cv::Mat> &list_t, const cv::Mat &K);

            double computeScoreForEH(double d2, double TM);

        private:
            // Keypoints from Reference Frame (Frame 1)
            std::vector<cv::KeyPoint> mvKeys1;

            // Keypoints from Current Frame (Frame 2)
            std::vector<cv::KeyPoint> mvKeys2;

            // Current Matches from Reference to Current
            std::vector<Match> mvMatches12;
            std::vector<bool> mvbMatched1;

            // Calibration
            cv::Mat mK;

            // Standard Deviation and Variance
            float mSigma, mSigma2;

            // Ransac max iterations
            int mMaxIterations;

            // Ransac sets
            std::vector<std::vector<size_t> > mvSets;
        };

    }
}


#endif //VI_SLAM_MOTION_ESTIMATION_H

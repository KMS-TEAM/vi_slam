//
// Created by lacie on 25/05/2021.
//

#ifndef VI_SLAM_FRAME_H
#define VI_SLAM_FRAME_H

#include "../common_include.h"
#include "vi_slam/basics/opencv_funcs.h"
#include "vi_slam/geometry/camera.h"
#include "vi_slam/geometry/feature_match.h"
#include "vi_slam/basics/converter.h"

#include "../../thirdparty/DBow3/DBoW3/src/BowVector.h"
#include "../../thirdparty/DBow3/DBoW3/src/FeatureVector.h"
#include "../../thirdparty/DBow3/DBoW3/src/Vocabulary.h"

namespace vi_slam{
    namespace vo{

#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

        typedef struct PtConn_
        {
            int pt_ref_idx;
            int pt_map_idx;
        } PtConn;

        class MapPoint;
        class KeyFrame;

        class Frame
        {
        public:
            typedef std::shared_ptr<Frame> Ptr;
            static int factory_id_;

        public:
            int id_;            // id of this frame
            int nNextId_;       // Next frame Id
            double time_stamp_; // when it is recorded

            // -- image features
            cv::Mat rgb_img_;
            vector<cv::KeyPoint> keypoints_;
            vector<cv::KeyPoint> ukeypoints_;
            cv::Mat descriptors_;
            vector<vector<unsigned char>> kpts_colors_; // rgb colors

            // Scale pyramid info.
            int mnScaleLevels;
            float mfScaleFactor;
            float mfLogScaleFactor;
            vector<float> mvScaleFactors;
            vector<float> mvInvScaleFactors;
            vector<float> mvLevelSigma2;
            vector<float> mvInvLevelSigma2;

            // Number of KeyPoints
            int N;

            // Corresponding stereo coordinate and depth for each keypoint.
            // "Monocular" keypoints have a negative value.
            std::vector<float> mvuRight;
            std::vector<float> mvDepth;

            // -- Vocabulary used for relocalization
            DBoW3::Vocabulary* mpVocaburary;

            // Bag of Words Vector structures.
            DBoW3::BowVector mBowVec;
            DBoW3::FeatureVector mFeatVec;

            // MapPoints associated to keypoints, NULL pointer if no association.
            std::vector<MapPoint*> mvpMapPoints;

            // -- Matches with reference keyframe (for E/H or PnP)
            //  for (1) E/H at initialization stage and (2) triangulating 3d points at all stages.
            vector<cv::DMatch> matches_with_ref_;         // matches with reference frame
            vector<cv::DMatch> inliers_matches_with_ref_; // matches that satisify E or H's constraints, and
            // Reference Keyframe.
            KeyFrame* mpReferenceKF;

            // -- vectors for triangulation
            vector<double> triangulation_angles_of_inliers_;
            vector<cv::DMatch> inliers_matches_for_3d_;                    // matches whose triangulation result is good.
            vector<cv::Point3f> inliers_pts3d_;                            // 3d points triangulated from inliers_matches_for_3d_
            std::unordered_map<int, PtConn> inliers_to_mappt_connections_; // curr idx -> idx in ref, and map

            // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
            static float mfGridElementWidthInv;
            static float mfGridElementHeightInv;
            std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

            // -- Matches with map points (for PnP)
            vector<cv::DMatch> matches_with_map_; // inliers matches index with respect to all the points

            // -- Camera
            geometry::Camera::Ptr camera_;

            // -- Current pose
            cv::Mat T_w_c_; // transform from world to camera

            // Undistorted Image Bounds (computed once).
            static float mnMinX;
            static float mnMaxX;
            static float mnMinY;
            static float mnMaxY;

            static bool mbInitialComputations;

        public:
            Frame() {}
            ~Frame() {}
            static Frame::Ptr createFrame(cv::Mat rgb_img, geometry::Camera::Ptr camera, double time_stamp = -1);

        public: // Below are deprecated. These were used in the two-frame-matching vo.
            void clearNoUsed()
            {
                // rgb_img_.release();
                kpts_colors_.clear();
                matches_with_ref_.clear();
                inliers_matches_with_ref_.clear();
                inliers_matches_for_3d_.clear();
                matches_with_map_.clear();
            }
            void calcKeyPoints()
            {
                geometry::calcKeyPoints(rgb_img_, keypoints_);
            }

            void calcDescriptors()
            {
                geometry::calcDescriptors(rgb_img_, keypoints_, descriptors_);
                kpts_colors_.clear();
                for (cv::KeyPoint kpt : keypoints_)
                {
                    int x = floor(kpt.pt.x), y = floor(kpt.pt.y);
                    kpts_colors_.push_back(basics::getPixelAt(rgb_img_, x, y));
                }
            };

            // Compute Bag of Words representation.
            void ComputeBoW();

            // Set the camera pose.
            void SetPose(cv::Mat Tcw);

            // Computes rotation, translation and camera center matrices from the camera pose.
            void UpdatePoseMatrices();

            // Returns the camera center.
            inline cv::Mat GetCameraCenter(){
                return mOw.clone();
            }

            // Returns inverse of rotation
            inline cv::Mat GetRotationInverse(){
                return mRwc.clone();
            }

            cv::Point2f projectWorldPointToImage(const cv::Point3f &p_world);
            bool isInFrame(const cv::Point3f &p_world);
            bool isInFrame(const cv::Mat &p_world);
            bool isMappoint(int idx)
            {
                bool not_find = inliers_to_mappt_connections_.find(idx) == inliers_to_mappt_connections_.end();
                return !not_find;
            }
            cv::Mat getCamCenter();

        private:

            // Undistort keypoints given OpenCV distortion parameters.
            // Only for the RGB-D case. Stereo must be already rectified!
            // (called in the constructor).
            void UndistortKeyPoints();

            // Computes image bounds for the undistorted image (called in the constructor).
            void ComputeImageBounds(const cv::Mat &im);

            // Assign keypoints to the grid for speed up feature matching (called in the constructor).
            void AssignFeaturesToGrid();

            // Rotation, translation and camera center
            cv::Mat mRcw;
            cv::Mat mtcw;
            cv::Mat mRwc;
            cv::Mat mOw; //==mtwc
        };
    }
}

#endif //VI_SLAM_FRAME_H
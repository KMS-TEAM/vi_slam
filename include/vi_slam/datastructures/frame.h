//
// Created by lacie on 25/05/2021.
//

#ifndef VI_SLAM_FRAME_H
#define VI_SLAM_FRAME_H

#include "vi_slam//common_include.h"
#include "vi_slam/basics/opencv_funcs.h"
#include "vi_slam/geometry/cameramodels/camera.h"
#include "vi_slam/geometry/fmatcher.h"
#include "vi_slam/basics/converter.h"
#include "vi_slam/geometry/fextractor.h"

#include "DBoW3/DBoW3/src/BowVector.h"
#include "DBoW3/DBoW3/src/FeatureVector.h"
#include "DBoW3/DBoW3/src/Vocabulary.h"

#include "vi_slam/datastructures/mappoint.h"
#include "vi_slam/datastructures/keyframe.h"
#include "vi_slam/datastructures/imu.h"

#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/slam/BetweenFactor.h>

#include <mutex>

namespace vi_slam{

    namespace geometry{
        class Camera;
        class FExtractor;
    }

    namespace optimization{
        class ConstraintPoseImu;
    }

    namespace datastructures{

    #define FRAME_GRID_ROWS 48
    #define FRAME_GRID_COLS 64

        class MapPoint;
        class KeyFrame;
        // class IMU;

        using namespace IMU;
        using namespace geometry;

        typedef struct PtConn_
        {
            int pt_ref_idx;
            int pt_map_idx;
        } PtConn;

        class Frame
        {
        public:
            typedef std::shared_ptr<Frame> Ptr;
            static int factory_id_;

        public:
            int id_;            // id of this frame
            static int nNextId_;       // Next frame Id
            double time_stamp_; // when it is recorded

            // -- image features
            cv::Mat rgb_img_;
            vector<cv::KeyPoint> keypoints_, keypointsRight_;
            vector<cv::KeyPoint> ukeypoints_;
            cv::Mat descriptors_, descriptorsRight_;
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
            // Feature extractor. The right is used only in the stereo case.
            geometry::FExtractor* mpORBextractorLeft, *mpORBextractorRight;

            // Bag of Words Vector structures.
            DBoW3::BowVector mBowVec;
            DBoW3::FeatureVector mFeatVec;

            // MapPoints associated to keypoints, NULL pointer if no association.
            std::vector<MapPoint*> mvpMapPoints;

            // Flag to identify outlier associations.
            std::vector<bool> mvbOutlier;
            int mnCloseMPs;

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
            geometry::Camera* camera_;

            // Calibration matrix and OpenCV distortion parameters.
            cv::Mat mK;
            static float fx;
            static float fy;
            static float cx;
            static float cy;
            static float invfx;
            static float invfy;
            cv::Mat mDistCoef;

            // Stereo baseline multiplied by fx.
            float mbf;

            // Stereo baseline in meters.
            float mb;

            // Threshold close/far points. Close points are inserted from 1 view.
            // Far points are inserted as in the monocular case from 2 views.
            float mThDepth;

            // -- Current pose
            cv::Mat T_w_c_; // transform from world to camera

            // Undistorted Image Bounds (computed once).
            static float mnMinX;
            static float mnMaxX;
            static float mnMinY;
            static float mnMaxY;

            static bool mbInitialComputations;

            // IMU
            // IMU linear velocity
            cv::Mat mVw;

            cv::Mat mPredRwb, mPredtwb, mPredVwb;
            IMU::Bias mPredBias;

            // IMU bias
            IMU::Bias mImuBias;

            // Imu calibration
            IMU::Calib mImuCalib;

            // Imu preintegration from last keyframe
            IMU::Preintegrated* mpImuPreintegrated;
            KeyFrame* mpLastKeyFrame;

            // Pointer to previous frame
            Frame* mpPrevFrame;
            IMU::Preintegrated* mpImuPreintegratedFrame;

            std::map<long unsigned int, cv::Point2f> mmProjectPoints;
            std::map<long unsigned int, cv::Point2f> mmMatchedInImage;

            string mNameFile;

            int mnDataset;

#ifdef REGISTER_TIMES
            double mTimeORB_Ext;
            double mTimeStereoMatch;
#endif

        public:
            Frame();
            ~Frame() {}
            static Frame::Ptr createFrame(cv::Mat rgb_img, geometry::Camera* camera, double time_stamp = -1);

            // Copy constructor.
            Frame(const Frame &frame);

            // Constructor for stereo cameras.
            Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, geometry::FExtractor* extractorLeft, geometry::FExtractor* extractorRight, DBoW3::Vocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, Camera* pCamera,Frame* pPrevF = static_cast<Frame*>(NULL), const IMU::Calib &ImuCalib = IMU::Calib());

            // Constructor for RGB-D cameras.
            Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, geometry::FExtractor* extractor,DBoW3::Vocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, Camera* pCamera,Frame* pPrevF = static_cast<Frame*>(NULL), const IMU::Calib &ImuCalib = IMU::Calib());

            // Constructor for Monocular cameras.
            Frame(const cv::Mat &imGray, const double &timeStamp, geometry::FExtractor* extractor,DBoW3::Vocabulary* voc, Camera* pCamera, cv::Mat &distCoef, const float &bf, const float &thDepth, Frame* pPrevF = static_cast<Frame*>(NULL), const IMU::Calib &ImuCalib = IMU::Calib());

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
                //geometry::calcKeyPoints(rgb_img_, keypoints_);
            }

            void calcDescriptors()
            {
//                geometry::calcDescriptors(rgb_img_, keypoints_, descriptors_);
//                kpts_colors_.clear();
//                for (cv::KeyPoint kpt : keypoints_)
//                {
//                    int x = floor(kpt.pt.x), y = floor(kpt.pt.y);
//                    kpts_colors_.push_back(basics::getPixelAt(rgb_img_, x, y));
//                }
            };

            // Extract ORB on the image. 0 for left image and 1 for right image.
            void ExtractORB(int flag, const cv::Mat &im, const int x0, const int x1);

            // Compute Bag of Words representation.
            void ComputeBoW();

            // Set the camera pose.
            void SetPose(cv::Mat Tcw);
            void GetPose(cv::Mat &Tcw);

            // Set IMU velocity
            void SetVelocity(const cv::Mat &Vwb);

            // Set IMU pose and velocity (implicitly changes camera pose)
            void SetImuPoseVelocity(const cv::Mat &Rwb, const cv::Mat &twb, const cv::Mat &Vwb);

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

            // Check if a MapPoint is in the frustum of the camera
            // and fill variables of the MapPoint to be used by the tracking
            bool isInFrustum(MapPoint* pMP, float viewingCosLimit);

            cv::Mat GetImuPosition();
            cv::Mat GetImuRotation();
            cv::Mat GetImuPose();

            void SetNewBias(const IMU::Bias &b);

            bool ProjectPointDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v);

            cv::Mat inRefCoordinates(cv::Mat pCw);

            // Compute the cell of a keypoint (return false if outside the grid)
            bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

            vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel=-1, const int maxLevel=-1, const bool bRight = false) const;

            // Search a match for each keypoint in the left image to a keypoint in the right image.
            // If there is a match, depth is computed and the right coordinate associated to the left keypoint is stored.
            void ComputeStereoMatches();

            // Associate a "right" coordinate to a keypoint if there is valid depth in the depthmap.
            void ComputeStereoFromRGBD(const cv::Mat &imDepth);

            // Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
            cv::Mat UnprojectStereo(const int &i);

            cv::Point2f projectWorldPointToImage(const cv::Point3f &p_world);
            bool isInFrame(const cv::Point3f &p_world);
            bool isInFrame(const cv::Mat &p_world);
            bool isMappoint(int idx)
            {
                bool not_find = inliers_to_mappt_connections_.find(idx) == inliers_to_mappt_connections_.end();
                return !not_find;
            }
            cv::Mat getCamCenter();

            optimization::ConstraintPoseImu* mpcpi;

            bool imuIsPreintegrated();
            void setIntegrated();

        public:

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

            cv::Matx31f mOwx;
            cv::Matx33f mRcwx;
            cv::Matx31f mtcwx;

            bool mbImuPreintegrated;

            std::mutex *mpMutexImu;

        public:
            geometry::Camera* mpCamera, *mpCamera2;

            //Number of KeyPoints extracted in the left and right images
            int Nleft, Nright;
            //Number of Non Lapping Keypoints
            int monoLeft, monoRight;

            //For stereo matching
            std::vector<int> mvLeftToRightMatch, mvRightToLeftMatch;

            //For stereo fisheye matching
            static cv::BFMatcher BFmatcher;

            //Triangulated stereo observations using as reference the left camera. These are
            //computed during ComputeStereoFishEyeMatches
            std::vector<cv::Mat> mvStereo3Dpoints;

            //Grid for the right image
            std::vector<std::size_t> mGridRight[FRAME_GRID_COLS][FRAME_GRID_ROWS];

            cv::Mat mTlr, mRlr, mtlr, mTrl;
            cv::Matx34f mTrlx, mTlrx;

            Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, FExtractor* extractorLeft, FExtractor* extractorRight, DBoW3::Vocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, Camera* pCamera, Camera* pCamera2, cv::Mat& Tlr,Frame* pPrevF = static_cast<Frame*>(NULL), const IMU::Calib &ImuCalib = IMU::Calib());

            //Stereo fisheye
            void ComputeStereoFishEyeMatches();

            bool isInFrustumChecks(MapPoint* pMP, float viewingCosLimit, bool bRight = false);

            cv::Mat UnprojectStereoFishEye(const int &i);

            cv::Mat imgLeft, imgRight;

            void PrintPointDistribution(){
                int left = 0, right = 0;
                int Nlim = (Nleft != -1) ? Nleft : N;
                for(int i = 0; i < N; i++){
                    if(mvpMapPoints[i] && !mvbOutlier[i]){
                        if(i < Nlim) left++;
                        else right++;
                    }
                }
                cout << "Point distribution in Frame: left-> " << left << " --- right-> " << right << endl;
            }
        };
    }
}

#endif //VI_SLAM_FRAME_H

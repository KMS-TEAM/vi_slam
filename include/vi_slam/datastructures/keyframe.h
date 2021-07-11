//
// Created by lacie on 05/06/2021.
//

#ifndef VI_SLAM_KEYFRAME_H
#define VI_SLAM_KEYFRAME_H

#include "DBoW3/DBoW3/src/BowVector.h"
#include "DBoW3/DBoW3/src/FeatureVector.h"
#include "DBoW3/DBoW3/src/Vocabulary.h"

#include "vi_slam/datastructures/frame.h"
#include "vi_slam/datastructures/keyframedatabase.h"
#include "vi_slam/datastructures/mappoint.h"
#include "vi_slam/datastructures/map.h"
#include "vi_slam/datastructures/imu.h"

#include "vi_slam/geometry/cameramodels/camera.h"
#include "vi_slam/geometry/fextractor.h"

#include "vi_slam/common_include.h"

#include <mutex>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>

namespace vi_slam{

    namespace geometry{
        class Camera;
    }

    namespace datastructures{

        class KeyFrameDatabase;
        class Map;
        class MapPoint;

        class KeyFrame {

        public:
            KeyFrame();
            KeyFrame(Frame &F, Map* pMap, KeyFrameDatabase* pKFDB);

            // Pose functions
            void SetPose(const cv::Mat &Tcw);
            void SetVelocity(const cv::Mat &Vw_);

            cv::Mat GetPose();
            cv::Mat GetPoseInverse();
            cv::Mat GetCameraCenter();
            cv::Mat GetStereoCenter();
            cv::Mat GetRotation();
            cv::Mat GetTranslation();

            cv::Mat GetImuPosition();
            cv::Mat GetImuRotation();
            cv::Mat GetImuPose();
            cv::Mat GetVelocity();

            cv::Matx33f GetRotation_();
            cv::Matx31f GetTranslation_();
            cv::Matx31f GetCameraCenter_();
            cv::Matx33f GetRightRotation_();
            cv::Matx31f GetRightTranslation_();
            cv::Matx44f GetRightPose_();
            cv::Matx31f GetRightCameraCenter_();
            cv::Matx44f GetPose_();

            // Bag of Words Representation
            void ComputeBoW();

            // Covisibility graph functions
            void AddConnection(KeyFrame* pKF, const int &weight);
            void EraseConnection(KeyFrame* pKF);
            void UpdateConnections(bool upParent = true);
            void UpdateBestCovisibles();
            std::set<KeyFrame *> GetConnectedKeyFrames();
            std::vector<KeyFrame* > GetVectorCovisibleKeyFrames();
            std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(const int &N);
            std::vector<KeyFrame*> GetCovisiblesByWeight(const int &w);
            int GetWeight(KeyFrame* pKF);

            // Spanning tree functions
            void AddChild(KeyFrame* pKF);
            void EraseChild(KeyFrame* pKF);
            void ChangeParent(KeyFrame* pKF);
            std::set<KeyFrame*> GetChilds();
            KeyFrame* GetParent();
            bool hasChild(KeyFrame* pKF);
            void SetFirstConnection(bool bFirst);

            // Loop Edges
            void AddLoopEdge(KeyFrame* pKF);
            std::set<KeyFrame*> GetLoopEdges();

            // Merge Edges
            void AddMergeEdge(KeyFrame* pKF);
            std::set<KeyFrame*> GetMergeEdges();

            // MapPoint observation functions
            int GetNumberMPs();
            void AddMapPoint(MapPoint* pMP, const size_t &idx);
            void EraseMapPointMatch(const size_t &idx);
            void EraseMapPointMatch(MapPoint* pMP);
            void ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP);
            std::set<MapPoint*> GetMapPoints();
            std::vector<MapPoint*> GetMapPointMatches();
            int TrackedMapPoints(const int &minObs);
            MapPoint* GetMapPoint(const size_t &idx);

            // KeyPoint functions
            std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const bool bRight = false) const;
            cv::Mat UnprojectStereo(int i);
            cv::Matx31f UnprojectStereo_(int i);

            // Image
            bool IsInImage(const float &x, const float &y) const;

            // Enable/Disable bad flag changes
            void SetNotErase();
            void SetErase();

            // Set/check bad flag
            void SetBadFlag();
            bool isBad();

            // Compute Scene Depth (q=2 median). Used in monocular.
            float ComputeSceneMedianDepth(const int q);

            static bool weightComp( int a, int b){
                return a>b;
            }

            static bool lId(KeyFrame* pKF1, KeyFrame* pKF2){
                return pKF1->mnId<pKF2->mnId;
            }

            Map* GetMap();
            void UpdateMap(Map* pMap);

            void SetNewBias(const IMU::Bias &b);
            cv::Mat GetGyroBias();
            cv::Mat GetAccBias();
            IMU::Bias GetImuBias();

            bool ProjectPointDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v);
            bool ProjectPointUnDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v);

            void SetORBVocabulary(DBoW3::Vocabulary* pORBVoc);
            void SetKeyFrameDatabase(KeyFrameDatabase* pKFDB);

            bool bImu;

            // The following variables are accesed from only 1 thread or never change (no mutex needed).
        public:

            static long unsigned int nNextId;
            long unsigned int mnId;
            const long unsigned int mnFrameId;

            const double mTimeStamp;

            // Grid (to speed up feature matching)
            const int mnGridCols;
            const int mnGridRows;
            const float mfGridElementWidthInv;
            const float mfGridElementHeightInv;

            // Variables used by the tracking
            long unsigned int mnTrackReferenceForFrame;
            long unsigned int mnFuseTargetForKF;

            // Variables used by the local mapping
            long unsigned int mnBALocalForKF;
            long unsigned int mnBAFixedForKF;

            //Number of optimizations by BA(amount of iterations in BA)
            long unsigned int mnNumberOfOpt;

            // Variables used by the keyframe database
            long unsigned int mnLoopQuery;
            int mnLoopWords;
            float mLoopScore;
            long unsigned int mnRelocQuery;
            int mnRelocWords;
            float mRelocScore;

            long unsigned int mnMergeQuery;
            int mnMergeWords;
            float mMergeScore;
            long unsigned int mnPlaceRecognitionQuery;
            int mnPlaceRecognitionWords;
            float mPlaceRecognitionScore;

            bool mbCurrentPlaceRecognition;

            // Variables used by loop closing
            cv::Mat mTcwGBA;
            cv::Mat mTcwBefGBA;
            cv::Mat mVwbGBA;
            cv::Mat mVwbBefGBA;
            IMU::Bias mBiasGBA;
            long unsigned int mnBAGlobalForKF;

            // Variables used by merging
            cv::Mat mTcwMerge;
            cv::Mat mTcwBefMerge;
            cv::Mat mTwcBefMerge;
            cv::Mat mVwbMerge;
            cv::Mat mVwbBefMerge;
            IMU::Bias mBiasMerge;
            long unsigned int mnMergeCorrectedForKF;
            long unsigned int mnMergeForKF;
            float mfScaleMerge;
            long unsigned int mnBALocalForMerge;

            float mfScale;

            // Calibration parameters
            const float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth;
            cv::Mat mDistCoef;

            // Number of KeyPoints
            const int N;

            // KeyPoints, stereo coordinate and descriptors (all associated by an index)
            const std::vector<cv::KeyPoint> mvKeys;
            const std::vector<cv::KeyPoint> mvKeysUn;
            const std::vector<float> mvuRight; // negative value for monocular points
            const std::vector<float> mvDepth; // negative value for monocular points
            const cv::Mat mDescriptors;

            //BoW
            DBoW3::BowVector mBowVec;
            DBoW3::FeatureVector mFeatVec;

            // Pose relative to parent (this is computed when bad flag is activated)
            cv::Mat mTcp;

            // Scale
            const int mnScaleLevels;
            const float mfScaleFactor;
            const float mfLogScaleFactor;
            const std::vector<float> mvScaleFactors;
            const std::vector<float> mvLevelSigma2;
            const std::vector<float> mvInvLevelSigma2;

            // Image bounds and calibration
            const int mnMinX;
            const int mnMinY;
            const int mnMaxX;
            const int mnMaxY;
            const cv::Mat mK;

            // Preintegrated IMU measurements from previous keyframe
            KeyFrame* mPrevKF;
            KeyFrame* mNextKF;

            IMU::Preintegrated* mpImuPreintegrated;
            IMU::Calib mImuCalib;


            unsigned int mnOriginMapId;

            string mNameFile;

            int mnDataset;

            std::vector <KeyFrame*> mvpLoopCandKFs;
            std::vector <KeyFrame*> mvpMergeCandKFs;

            bool mbHasHessian;
            cv::Mat mHessianPose;

            // The following variables need to be accessed trough a mutex to be thread safe.
        protected:

            // SE3 Pose and camera center
            cv::Mat Tcw;
            cv::Mat Twc;
            cv::Mat Ow;
            cv::Mat Cw; // Stereo middel point. Only for visualization

            cv::Matx44f Tcw_, Twc_, Tlr_;
            cv::Matx31f Ow_;

            // IMU position
            cv::Mat Owb;

            // Velocity (Only used for inertial SLAM)
            cv::Mat Vw;

            // Imu bias
            IMU::Bias mImuBias;

            // MapPoints associated to keypoints
            std::vector<MapPoint*> mvpMapPoints;

            // BoW
            KeyFrameDatabase* mpKeyFrameDB;
            DBoW3::Vocabulary* mpORBvocabulary;

            // Grid over the image to speed up feature matching
            std::vector< std::vector <std::vector<size_t> > > mGrid;

            std::map<KeyFrame*,int> mConnectedKeyFrameWeights;
            std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames;
            std::vector<int> mvOrderedWeights;

            // Spanning Tree and Loop Edges
            bool mbFirstConnection;
            KeyFrame* mpParent;
            std::set<KeyFrame*> mspChildrens;
            std::set<KeyFrame*> mspLoopEdges;
            std::set<KeyFrame*> mspMergeEdges;

            // Bad flags
            bool mbNotErase;
            bool mbToBeErased;
            bool mbBad;

            float mHalfBaseline; // Only for visualization

            Map* mpMap;

            std::mutex mMutexPose;
            std::mutex mMutexConnections;
            std::mutex mMutexFeatures;
            std::mutex mMutexMap;

        public:
            geometry::Camera* mpCamera, *mpCamera2;

            //Indexes of stereo observations correspondences
            std::vector<int> mvLeftToRightMatch, mvRightToLeftMatch;

            //Transformation matrix between cameras in stereo fisheye
            cv::Mat mTlr;
            cv::Mat mTrl;

            //KeyPoints in the right image (for stereo fisheye, coordinates are needed)
            const std::vector<cv::KeyPoint> mvKeysRight;

            const int NLeft, NRight;

            std::vector< std::vector <std::vector<size_t> > > mGridRight;

            cv::Mat GetRightPose();
            cv::Mat GetRightPoseInverse();
            cv::Mat GetRightPoseInverseH();
            cv::Mat GetRightCameraCenter();
            cv::Mat GetRightRotation();
            cv::Mat GetRightTranslation();

            cv::Mat imgLeft, imgRight;

            void PrintPointDistribution(){
                int left = 0, right = 0;
                int Nlim = (NLeft != -1) ? NLeft : N;
                for(int i = 0; i < N; i++){
                    if(mvpMapPoints[i]){
                        if(i < Nlim) left++;
                        else right++;
                    }
                }
                cout << "Point distribution in KeyFrame: left-> " << left << " --- right-> " << right << endl;
            }
        };
    }
}

#endif //VI_SLAM_KEYFRAME_H

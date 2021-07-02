//
// Created by lacie on 25/05/2021.
//

#ifndef VI_SLAM_MAPPOINT_H
#define VI_SLAM_MAPPOINT_H

#include "vi_slam/common_include.h"
#include "vi_slam/datastructures/frame.h"
#include "vi_slam/datastructures/keyframe.h"
#include "vi_slam/datastructures/map.h"

#include<mutex>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>

namespace vi_slam{
    namespace datastructures{

        class KeyFrame;
        class Map;
        class Frame;

        class MapPoint
        {
        public: // Basics Properties

            typedef std::shared_ptr<MapPoint> Ptr;

            // Id
            static int factory_id_;
            int id_;
            static long unsigned int nNextId;
            long int mnFirstKFid;
            long int mnFirstFrame;
            int nObs;

            Map* mpMap;

            // Pose informations
            cv::Mat pos_;
            cv::Mat norm_;                    // Vector pointing from camera center to the point
            vector<unsigned char> color_; // r,g,b
            cv::Mat descriptor_;              // Descriptor for matching

            // Position in absolute coordinates
            cv::Mat mWorldPos;
            cv::Matx31f mWorldPosx;

            // Keyframes observing the point and associated index in keyframe
            std::map<KeyFrame*,std::tuple<int,int> > mObservations;
            // Mean viewing direction
            cv::Mat mNormalVector;
            cv::Matx31f mNormalVectorx;

            // Reference KeyFrame
            KeyFrame* mpRefKF;

            // Tracking counters
            int mnVisible;
            int mnFound;

            // Bad flag (we do not currently erase MapPoint from memory)
            bool mbBad;
            MapPoint* mpReplaced;

            // Scale invariance distances
            float mfMinDistance;
            float mfMaxDistance;

            // Variables used by the tracking
            float mTrackProjX;
            float mTrackProjY;
            float mTrackDepth;
            float mTrackDepthR;
            float mTrackProjXR;
            float mTrackProjYR;
            bool mbTrackInView, mbTrackInViewR;
            int mnTrackScaleLevel, mnTrackScaleLevelR;
            float mTrackViewCos, mTrackViewCosR;
            long unsigned int mnTrackReferenceForFrame;
            long unsigned int mnLastFrameSeen;

            // Variables used by local mapping
            long unsigned int mnBALocalForKF;
            long unsigned int mnFuseCandidateForKF;

            // Variables used by loop closing
            long unsigned int mnLoopPointForKF;
            long unsigned int mnCorrectedByKF;
            long unsigned int mnCorrectedReference;
            cv::Mat mPosGBA;
            long unsigned int mnBAGlobalForKF;
            long unsigned int mnBALocalForMerge;

            // Variable used by merging
            cv::Mat mPosMerge;
            cv::Mat mNormalVectorMerge;

            static std::mutex mGlobalMutex;
            std::mutex mMutexPos;
            std::mutex mMutexFeatures;
            std::mutex mMutexMap;

            // Fopr inverse depth optimization
            double mInvDepth;
            double mInitU;
            double mInitV;
            KeyFrame* mpHostKF;

            unsigned int mnOriginMapId;

        public:                 // Properties for constructing local mapping
            bool good_;         // TODO: determine wheter a good point
            int matched_times_; // being an inliner in pose estimation
            int visible_times_; // being visible in current frame

        public: // Functions
            MapPoint();
            MapPoint(const cv::Point3f &pos, const cv::Mat &descriptor, const cv::Mat &norm,
                     unsigned char r = 0, unsigned char g = 0, unsigned char b = 0);

            MapPoint(const cv::Mat &Pos, KeyFrame* pRefKF, Map* pMap);
            MapPoint(const cv::Mat &Pos,  Map* pMap, Frame* pFrame, const int &idxF);
            MapPoint(const double invDepth, cv::Point2f uv_init, KeyFrame* pRefKF, KeyFrame* pHostKF, Map* pMap);

            void SetWorldPos(const cv::Mat &Pos);
            cv::Mat GetWorldPos();

            cv::Mat GetNormal();

            cv::Matx31f GetWorldPos2();
            cv::Matx31f GetNormal2();

            KeyFrame* GetReferenceKeyFrame();

            std::map<KeyFrame*,size_t> GetObservations();
            int Observations();

            void AddObservation(KeyFrame* pKF,size_t idx);
            void EraseObservation(KeyFrame* pKF);

            std::tuple<int,int> GetIndexInKeyFrame(KeyFrame* pKF);
            bool IsInKeyFrame(KeyFrame* pKF);

            void SetBadFlag();
            bool isBad();

            void Replace(MapPoint* pMP);
            MapPoint* GetReplaced();

            void IncreaseVisible(int n=1);
            void IncreaseFound(int n=1);
            float GetFoundRatio();
            inline int GetFound(){
                return mnFound;
            }

            void ComputeDistinctiveDescriptors();

            cv::Mat GetDescriptor();

            void UpdateNormalAndDepth();
            void SetNormalVector(cv::Mat& normal);

            float GetMinDistanceInvariance();
            float GetMaxDistanceInvariance();
            int PredictScale(const float &currentDist, KeyFrame*pKF);
            int PredictScale(const float &currentDist, Frame* pF);

            Map* GetMap();
            void UpdateMap(Map* pMap);
        };
    }
}

#endif //VI_SLAM_MAPPOINT_H

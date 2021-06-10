//
// Created by lacie on 25/05/2021.
//

#include "vi_slam/vo/mappoint.h"
#include "vi_slam/common_include.h"
#include "vi_slam/basics/opencv_funcs.h"
#include <iostream>
#include <mutex>

using namespace std;

namespace vi_slam{
    namespace vo{
        int MapPoint::factory_id_ = 0;
        long unsigned int MapPoint::nNextId=0;
        std::mutex MapPoint::mGlobalMutex;

        MapPoint::MapPoint(
                const cv::Point3f &pos, const cv::Mat &descriptor, const cv::Mat &norm,
                unsigned char r, unsigned char g, unsigned char b) : pos_(pos), descriptor_(descriptor), norm_(norm), color_({r, g, b}),
                                                                     good_(true), visible_times_(1), matched_times_(1)

        {
            id_ = factory_id_++;
        }

        void MapPoint::setPos(const cv::Mat &Pos){
            std::unique_lock<std::mutex> lock2(mGlobalMutex);
            std::unique_lock<std::mutex> lock(mMutexPos);
            pos_ = Pos.clone();
        }

        MapPoint::MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map* pMap):
                mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
                mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
                mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
                mpReplaced(static_cast<MapPoint*>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap)
        {
            pos_ = Pos.clone();
            norm_ = cv::Mat::zeros(3,1,CV_32F);

            // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
            std::unique_lock<std::mutex> lock(mpMap->mMutexPointCreation);
            id_=nNextId++;
        }

        MapPoint::MapPoint(const cv::Mat &Pos, Map* pMap, Frame* pFrame, const int &idxF):
                mnFirstKFid(-1), mnFirstFrame(pFrame->id_), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
                mnBALocalForKF(0), mnFuseCandidateForKF(0),mnLoopPointForKF(0), mnCorrectedByKF(0),
                mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(static_cast<KeyFrame*>(NULL)), mnVisible(1),
                mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap)
        {
            pos_ = Pos.clone();
            cv::Mat Ow = pFrame->GetCameraCenter();
            norm_ = vi_slam::basics::point3f_to_mat3x1(pos_) - Ow;
            norm_ = norm_/cv::norm(norm_);

            cv::Mat PC = Pos - Ow;
            const float dist = cv::norm(PC);
            const int level = pFrame->ukeypoints_[idxF].octave;
            const float levelScaleFactor =  pFrame->mvScaleFactors[level];
            const int nLevels = pFrame->mnScaleLevels;

            mfMaxDistance = dist*levelScaleFactor;
            mfMinDistance = mfMaxDistance/pFrame->mvScaleFactors[nLevels-1];

            pFrame->descriptors_.row(idxF).copyTo(descriptor_);

            // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
            std::unique_lock<std::mutex> lock(mpMap->mMutexPointCreation);
            id_=nNextId++;
        }

        cv::Mat MapPoint::GetWorldPos()
        {
            std::unique_lock<std::mutex> lock(mMutexPos);
            return pos_.clone();
        }

        cv::Mat MapPoint::GetNormal()
        {
            unique_lock<mutex> lock(mMutexPos);
            return norm_.clone();
        }

        KeyFrame* MapPoint::GetReferenceKeyFrame()
        {
            unique_lock<mutex> lock(mMutexFeatures);
            return mpRefKF;
        }

        void MapPoint::AddObservation(KeyFrame* pKF, size_t idx)
        {
            unique_lock<mutex> lock(mMutexFeatures);
            if(mObservations.count(pKF))
                return;
            mObservations[pKF]=idx;

            if(pKF->mvuRight[idx]>=0)
                nObs+=2;
            else
                nObs++;
        }

        void MapPoint::EraseObservation(KeyFrame* pKF)
        {
            bool bBad=false;
            {
                unique_lock<mutex> lock(mMutexFeatures);
                if(mObservations.count(pKF))
                {
                    int idx = mObservations[pKF];
                    if(pKF->mvuRight[idx]>=0)
                        nObs-=2;
                    else
                        nObs--;

                    mObservations.erase(pKF);

                    if(mpRefKF==pKF)
                        mpRefKF=mObservations.begin()->first;

                    // If only 2 observations or less, discard point
                    if(nObs<=2)
                        bBad=true;
                }
            }

            if(bBad)
                SetBadFlag();
        }

        map<KeyFrame*, size_t> MapPoint::GetObservations()
        {
            unique_lock<mutex> lock(mMutexFeatures);
            return mObservations;
        }

        int MapPoint::Observations()
        {
            unique_lock<mutex> lock(mMutexFeatures);
            return nObs;
        }

        void MapPoint::SetBadFlag()
        {
            map<KeyFrame*,size_t> obs;
            {
                unique_lock<mutex> lock1(mMutexFeatures);
                unique_lock<mutex> lock2(mMutexPos);
                mbBad=true;
                obs = mObservations;
                mObservations.clear();
            }
            for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
            {
                KeyFrame* pKF = mit->first;
                pKF->EraseMapPointMatch(mit->second);
            }

            mpMap->EraseMapPoint(this);
        }

        MapPoint* MapPoint::GetReplaced()
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            return mpReplaced;
        }

        void MapPoint::Replace(MapPoint* pMP)
        {
            if(pMP->id_==this->id_)
                return;

            int nvisible, nfound;
            map<KeyFrame*,size_t> obs;
            {
                unique_lock<mutex> lock1(mMutexFeatures);
                unique_lock<mutex> lock2(mMutexPos);
                obs=mObservations;
                mObservations.clear();
                mbBad=true;
                nvisible = mnVisible;
                nfound = mnFound;
                mpReplaced = pMP;
            }

            for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
            {
                // Replace measurement in keyframe
                KeyFrame* pKF = mit->first;

                if(!pMP->IsInKeyFrame(pKF))
                {
                    pKF->ReplaceMapPointMatch(mit->second, pMP);
                    pMP->AddObservation(pKF,mit->second);
                }
                else
                {
                    pKF->EraseMapPointMatch(mit->second);
                }
            }
            pMP->IncreaseFound(nfound);
            pMP->IncreaseVisible(nvisible);
            pMP->ComputeDistinctiveDescriptors();

            mpMap->EraseMapPoint(this);
        }

        bool MapPoint::isBad()
        {
            unique_lock<mutex> lock(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            return mbBad;
        }

        void MapPoint::IncreaseVisible(int n)
        {
            unique_lock<mutex> lock(mMutexFeatures);
            mnVisible+=n;
        }

        void MapPoint::IncreaseFound(int n)
        {
            unique_lock<mutex> lock(mMutexFeatures);
            mnFound+=n;
        }

        float MapPoint::GetFoundRatio()
        {
            unique_lock<mutex> lock(mMutexFeatures);
            return static_cast<float>(mnFound)/mnVisible;
        }

        void MapPoint::ComputeDistinctiveDescriptors()
        {
            // Retrieve all observed descriptors
            vector<cv::Mat> vDescriptors;

            map<KeyFrame*,size_t> observations;

            {
                unique_lock<mutex> lock1(mMutexFeatures);
                if(mbBad)
                    return;
                observations=mObservations;
            }

            if(observations.empty())
                return;

            vDescriptors.reserve(observations.size());

            for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
            {
                KeyFrame* pKF = mit->first;

                if(!pKF->isBad())
                    vDescriptors.push_back(pKF->mDescriptors.row(mit->second));
            }

            if(vDescriptors.empty())
                return;

            // Compute distances between them
            const size_t N = vDescriptors.size();

            float Distances[N][N];
            for(size_t i=0;i<N;i++)
            {
                Distances[i][i]=0;
                for(size_t j=i+1;j<N;j++)
                {
                    int distij = ORBmatcher::DescriptorDistance(vDescriptors[i],vDescriptors[j]);
                    Distances[i][j]=distij;
                    Distances[j][i]=distij;
                }
            }

            // Take the descriptor with least median distance to the rest
            int BestMedian = INT_MAX;
            int BestIdx = 0;
            for(size_t i=0;i<N;i++)
            {
                vector<int> vDists(Distances[i],Distances[i]+N);
                sort(vDists.begin(),vDists.end());
                int median = vDists[0.5*(N-1)];

                if(median<BestMedian)
                {
                    BestMedian = median;
                    BestIdx = i;
                }
            }

            {
                unique_lock<mutex> lock(mMutexFeatures);
                descriptor_ = vDescriptors[BestIdx].clone();
            }
        }

        cv::Mat MapPoint::GetDescriptor()
        {
            unique_lock<mutex> lock(mMutexFeatures);
            return descriptor_.clone();
        }

        int MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
        {
            unique_lock<mutex> lock(mMutexFeatures);
            if(mObservations.count(pKF))
                return mObservations[pKF];
            else
                return -1;
        }

        bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
        {
            unique_lock<mutex> lock(mMutexFeatures);
            return (mObservations.count(pKF));
        }

        void MapPoint::UpdateNormalAndDepth()
        {
            map<KeyFrame*,size_t> observations;
            KeyFrame* pRefKF;
            cv::Mat Pos;
            {
                unique_lock<mutex> lock1(mMutexFeatures);
                unique_lock<mutex> lock2(mMutexPos);
                if(mbBad)
                    return;
                observations=mObservations;
                pRefKF=mpRefKF;
                Pos = pos_.clone();
            }

            if(observations.empty())
                return;

            cv::Mat normal = cv::Mat::zeros(3,1,CV_32F);
            int n=0;
            for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
            {
                KeyFrame* pKF = mit->first;
                cv::Mat Owi = pKF->GetCameraCenter();
                cv::Mat normali = pos_ - Owi;
                normal = normal + normali/cv::norm(normali);
                n++;
            }

            cv::Mat PC = Pos - pRefKF->GetCameraCenter();
            const float dist = cv::norm(PC);
            const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
            const float levelScaleFactor =  pRefKF->mvScaleFactors[level];
            const int nLevels = pRefKF->mnScaleLevels;

            {
                unique_lock<mutex> lock3(mMutexPos);
                mfMaxDistance = dist*levelScaleFactor;
                mfMinDistance = mfMaxDistance/pRefKF->mvScaleFactors[nLevels-1];
                norm_ = normal/n;
            }
        }

        float MapPoint::GetMinDistanceInvariance()
        {
            unique_lock<mutex> lock(mMutexPos);
            return 0.8f*mfMinDistance;
        }

        float MapPoint::GetMaxDistanceInvariance()
        {
            unique_lock<mutex> lock(mMutexPos);
            return 1.2f*mfMaxDistance;
        }

        int MapPoint::PredictScale(const float &currentDist, KeyFrame* pKF)
        {
            float ratio;
            {
                unique_lock<mutex> lock(mMutexPos);
                ratio = mfMaxDistance/currentDist;
            }

            int nScale = ceil(log(ratio)/pKF->mfLogScaleFactor);
            if(nScale<0)
                nScale = 0;
            else if(nScale>=pKF->mnScaleLevels)
                nScale = pKF->mnScaleLevels-1;

            return nScale;
        }

        int MapPoint::PredictScale(const float &currentDist, Frame* pF)
        {
            float ratio;
            {
                unique_lock<mutex> lock(mMutexPos);
                ratio = mfMaxDistance/currentDist;
            }

            int nScale = ceil(log(ratio)/pF->mfLogScaleFactor);
            if(nScale<0)
                nScale = 0;
            else if(nScale>=pF->mnScaleLevels)
                nScale = pF->mnScaleLevels-1;

            return nScale;
        }

    }
}

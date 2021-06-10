//
// Created by lacie on 25/05/2021.
//

#include "vi_slam/datastructures/map.h"
#include <mutex>

using namespace std;

namespace vi_slam{
    namespace vo{

        Map::Map():mnMaxKFid(0),mnBigChangeIdx(0)
        {
        }

        void Map::insertKeyFrame(Frame::Ptr frame)
        {
            if (keyframes_.find(frame->id_) == keyframes_.end())
            {
                keyframes_.insert(make_pair(frame->id_, frame));
            }
            else
            {
                keyframes_[frame->id_] = frame;
            }
            printf("Insert keyframe!!! frame_id = %d, total keyframes = %d\n", frame->id_, (int)keyframes_.size());
        }

        void Map::insertMapPoint(MapPoint::Ptr map_point)
        {
            if (map_points_.find(map_point->id_) == map_points_.end())
            {
                map_points_.insert(make_pair(map_point->id_, map_point));
            }
            else
            {
                map_points_[map_point->id_] = map_point;
            }
        }

        Frame::Ptr Map::findKeyFrame(int frame_id)
        {
            if (keyframes_.find(frame_id) == keyframes_.end())
                return NULL;
            else
                return keyframes_[frame_id];
        }

        bool Map::hasKeyFrame(int frame_id)
        {
            if (keyframes_.find(frame_id) == keyframes_.end())
                return false;
            else
                return true;
        }

        void Map::EraseMapPoint(MapPoint *pMP)
        {
            unique_lock<mutex> lock(mMutexMap);
            mspMapPoints.erase(pMP);

            // TODO: This only erase the pointer.
            // Delete the MapPoint
        }

        void Map::EraseKeyFrame(KeyFrame *pKF)
        {
            unique_lock<mutex> lock(mMutexMap);
            mspKeyFrames.erase(pKF);

            // TODO: This only erase the pointer.
            // Delete the MapPoint
        }

        void Map::SetReferenceMapPoints(const vector<MapPoint *> &vpMPs)
        {
            unique_lock<mutex> lock(mMutexMap);
            mvpReferenceMapPoints = vpMPs;
        }

        void Map::InformNewBigChange()
        {
            unique_lock<mutex> lock(mMutexMap);
            mnBigChangeIdx++;
        }

        int Map::GetLastBigChangeIdx()
        {
            unique_lock<mutex> lock(mMutexMap);
            return mnBigChangeIdx;
        }

        vector<KeyFrame*> Map::GetAllKeyFrames()
        {
            unique_lock<mutex> lock(mMutexMap);
            return vector<KeyFrame*>(mspKeyFrames.begin(),mspKeyFrames.end());
        }

        vector<MapPoint*> Map::GetAllMapPoints()
        {
            unique_lock<mutex> lock(mMutexMap);
            return vector<MapPoint*>(mspMapPoints.begin(),mspMapPoints.end());
        }

        long unsigned int Map::MapPointsInMap()
        {
            unique_lock<mutex> lock(mMutexMap);
            return mspMapPoints.size();
        }

        long unsigned int Map::KeyFramesInMap()
        {
            unique_lock<mutex> lock(mMutexMap);
            return mspKeyFrames.size();
        }

        vector<MapPoint*> Map::GetReferenceMapPoints()
        {
            unique_lock<mutex> lock(mMutexMap);
            return mvpReferenceMapPoints;
        }

        long unsigned int Map::GetMaxKFid()
        {
            unique_lock<mutex> lock(mMutexMap);
            return mnMaxKFid;
        }

        void Map::clear()
        {
            for(set<MapPoint*>::iterator sit=mspMapPoints.begin(), send=mspMapPoints.end(); sit!=send; sit++)
                delete *sit;

            for(set<KeyFrame*>::iterator sit=mspKeyFrames.begin(), send=mspKeyFrames.end(); sit!=send; sit++)
                delete *sit;

            mspMapPoints.clear();
            mspKeyFrames.clear();
            mnMaxKFid = 0;
            mvpReferenceMapPoints.clear();
            mvpKeyFrameOrigins.clear();
        }
    }
}

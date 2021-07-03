//
// Created by lacie on 05/06/2021.
//

#ifndef VI_SLAM_KEYFRAMEDATABASE_H
#define VI_SLAM_KEYFRAMEDATABASE_H

#include <vector>
#include <list>
#include <set>

#include "vi_slam/common_include.h"
#include "vi_slam/datastructures/keyframe.h"
#include "vi_slam/datastructures/frame.h"
#include "vi_slam/datastructures/map.h"

#include "DBoW3/DBoW3/src/Vocabulary.h"

#include <mutex>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/list.hpp>

namespace vi_slam{
    namespace datastructures{

        class KeyFrame;
        class Frame;
        class Map;

        // using namespace std;
        class KeyFrameDatabase {
        public:

            KeyFrameDatabase(const DBoW3::Vocabulary &voc);

            void add(KeyFrame* pKF);

            void erase(KeyFrame* pKF);

            void clear();
            void clearMap(Map* pMap);

            // Loop Detection
            std::vector<KeyFrame *> DetectLoopCandidates(KeyFrame* pKF, float minScore);

            // Loop and Merge Detection
            void DetectCandidates(KeyFrame* pKF, float minScore,vector<KeyFrame*>& vpLoopCand, vector<KeyFrame*>& vpMergeCand);
            void DetectBestCandidates(KeyFrame *pKF, vector<KeyFrame*> &vpLoopCand, vector<KeyFrame*> &vpMergeCand, int nMinWords);
            void DetectNBestCandidates(KeyFrame *pKF, vector<KeyFrame*> &vpLoopCand, vector<KeyFrame*> &vpMergeCand, int nNumCandidates);

            // Relocalization
            std::vector<KeyFrame*> DetectRelocalizationCandidates(Frame* F, Map* pMap);

            void SetORBVocabulary(DBoW3::Vocabulary* pORBVoc);

        protected:

            // Associated vocabulary
            const DBoW3::Vocabulary* mpVoc;

            // Inverted file
            std::vector<std::list<KeyFrame*> > mvInvertedFile;

            // Mutex
            std::mutex mMutex;
        };
    }
}

#endif //VI_SLAM_KEYFRAMEDATABASE_H

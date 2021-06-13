//
// Created by lacie on 05/06/2021.
//

#ifndef VI_SLAM_KEYFRAMEDATABASE_H
#define VI_SLAM_KEYFRAMEDATABASE_H

#include <vector>
#include <list>
#include <set>

#include "vi_slam/datastructures/keyframe.h"
#include "vi_slam/datastructures/frame.h"
#include "DBow3/DBoW3/src/Vocabulary.h"
#include "keyframe.h"

#include <mutex>

namespace vi_slam{
    namespace datastructures{

        class KeyFrame;
        class Frame;

        // using namespace std;
        class KeyFrameDatabase {
        public:

            KeyFrameDatabase(const DBoW3::Vocabulary &voc);

            void add(datastructures::KeyFrame* pKF);

            void erase(datastructures::KeyFrame* pKF);

            void clear();

            // Loop Detection
            std::vector<datastructures::KeyFrame *> DetectLoopCandidates(datastructures::KeyFrame* pKF, float minScore);

            // Relocalization
            std::vector<datastructures::KeyFrame*> DetectRelocalizationCandidates(datastructures::Frame* F);

        protected:

            // Associated vocabulary
            const DBoW3::Vocabulary* mpVoc;

            // Inverted file
            std::vector<std::list<datastructures::KeyFrame*> > mvInvertedFile;

            // Mutex
            std::mutex mMutex;
        };
    }
}

#endif //VI_SLAM_KEYFRAMEDATABASE_H

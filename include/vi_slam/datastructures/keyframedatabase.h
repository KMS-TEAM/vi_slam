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

#include <mutex>

namespace vi_slam{
    namespace datastructures{

        class KeyFrame;
        class Frame;

        // using namespace std;
        class KeyFrameDatabase {
        public:

            KeyFrameDatabase(const DBoW3::Vocabulary &voc);

            void add(KeyFrame* pKF);

            void erase(KeyFrame* pKF);

            void clear();

            // Loop Detection
            std::vector<KeyFrame *> DetectLoopCandidates(KeyFrame* pKF, float minScore);

            // Relocalization
            std::vector<KeyFrame*> DetectRelocalizationCandidates(Frame* F);

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

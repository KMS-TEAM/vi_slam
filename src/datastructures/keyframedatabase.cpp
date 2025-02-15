//
// Created by lacie on 05/06/2021.
//

#include "vi_slam/datastructures/keyframedatabase.h"
#include "vi_slam/datastructures/keyframe.h"
#include "DBoW3/DBoW3/src/BowVector.h"
#include <mutex>

using namespace std;

namespace vi_slam{
    namespace datastructures{
        KeyFrameDatabase::KeyFrameDatabase (const DBoW3::Vocabulary &voc):
                mpVoc(&voc)
        {
            mvInvertedFile.resize(voc.size());
        }


        void KeyFrameDatabase::add(KeyFrame *pKF)
        {
            unique_lock<mutex> lock(mMutex);

            for(DBoW3::BowVector::const_iterator vit= pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
                mvInvertedFile[vit->first].push_back(pKF);
        }

        void KeyFrameDatabase::erase(KeyFrame* pKF)
        {
            unique_lock<mutex> lock(mMutex);

            // Erase elements in the Inverse File for the entry
            for(DBoW3::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
            {
                // List of keyframes that share the word
                list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

                for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
                {
                    if(pKF==*lit)
                    {
                        lKFs.erase(lit);
                        break;
                    }
                }
            }
        }

        void KeyFrameDatabase::clear()
        {
            mvInvertedFile.clear();
            mvInvertedFile.resize(mpVoc->size());
        }

        void KeyFrameDatabase::clearMap(Map* pMap)
        {
            unique_lock<mutex> lock(mMutex);

            // Erase elements in the Inverse File for the entry
            for(std::vector<list<KeyFrame*> >::iterator vit=mvInvertedFile.begin(), vend=mvInvertedFile.end(); vit!=vend; vit++)
            {
                // List of keyframes that share the word
                list<KeyFrame*> &lKFs =  *vit;

                for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend;)
                {
                    KeyFrame* pKFi = *lit;
                    if(pMap == pKFi->GetMap())
                    {
                        lit = lKFs.erase(lit);
                        // Dont delete the KF because the class Map clean all the KF when it is destroyed
                    }
                    else
                    {
                        ++lit;
                    }
                }
            }
        }

        vector<KeyFrame*> KeyFrameDatabase::DetectLoopCandidates(KeyFrame* pKF, float minScore)
        {
            set<KeyFrame*> spConnectedKeyFrames = pKF->GetConnectedKeyFrames();
            list<KeyFrame*> lKFsSharingWords;

            // Search all keyframes that share a word with current keyframes
            // Discard keyframes connected to the query keyframe
            {
                unique_lock<mutex> lock(mMutex);

                for(DBoW3::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit != vend; vit++)
                {
                    list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

                    for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
                    {
                        KeyFrame* pKFi=*lit;
                        if(pKFi->mnLoopQuery!=pKF->mnId)
                        {
                            pKFi->mnLoopWords=0;
                            if(!spConnectedKeyFrames.count(pKFi))
                            {
                                pKFi->mnLoopQuery=pKF->mnId;
                                lKFsSharingWords.push_back(pKFi);
                            }
                        }
                        pKFi->mnLoopWords++;
                    }
                }
            }

            if(lKFsSharingWords.empty())
                return vector<KeyFrame*>();

            list<pair<float,KeyFrame*> > lScoreAndMatch;

            // Only compare against those keyframes that share enough words
            int maxCommonWords=0;
            for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
            {
                if((*lit)->mnLoopWords>maxCommonWords)
                    maxCommonWords=(*lit)->mnLoopWords;
            }

            int minCommonWords = maxCommonWords*0.8f;

            int nscores=0;

            // Compute similarity score. Retain the matches whose score is higher than minScore
            for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi = *lit;

                if(pKFi->mnLoopWords>minCommonWords)
                {
                    nscores++;

                    float si = mpVoc->score(pKF->mBowVec,pKFi->mBowVec);

                    pKFi->mLoopScore = si;
                    if(si>=minScore)
                        lScoreAndMatch.push_back(make_pair(si,pKFi));
                }
            }

            if(lScoreAndMatch.empty())
                return vector<KeyFrame*>();

            list<pair<float,KeyFrame*> > lAccScoreAndMatch;
            float bestAccScore = minScore;

            // Lets now accumulate score by covisibility
            for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
            {
                KeyFrame* pKFi = it->second;
                vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

                float bestScore = it->first;
                float accScore = it->first;
                KeyFrame* pBestKF = pKFi;
                for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
                {
                    KeyFrame* pKF2 = *vit;
                    if(pKF2->mnLoopQuery==pKF->mnId && pKF2->mnLoopWords>minCommonWords)
                    {
                        accScore+=pKF2->mLoopScore;
                        if(pKF2->mLoopScore>bestScore)
                        {
                            pBestKF=pKF2;
                            bestScore = pKF2->mLoopScore;
                        }
                    }
                }

                lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
                if(accScore>bestAccScore)
                    bestAccScore=accScore;
            }

            // Return all those keyframes with a score higher than 0.75*bestScore
            float minScoreToRetain = 0.75f*bestAccScore;

            set<KeyFrame*> spAlreadyAddedKF;
            vector<KeyFrame*> vpLoopCandidates;
            vpLoopCandidates.reserve(lAccScoreAndMatch.size());

            for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
            {
                if(it->first>minScoreToRetain)
                {
                    KeyFrame* pKFi = it->second;
                    if(!spAlreadyAddedKF.count(pKFi))
                    {
                        vpLoopCandidates.push_back(pKFi);
                        spAlreadyAddedKF.insert(pKFi);
                    }
                }
            }
            return vpLoopCandidates;
        }

        void KeyFrameDatabase::DetectCandidates(KeyFrame* pKF, float minScore,vector<KeyFrame*>& vpLoopCand, vector<KeyFrame*>& vpMergeCand)
        {
            set<KeyFrame*> spConnectedKeyFrames = pKF->GetConnectedKeyFrames();
            list<KeyFrame*> lKFsSharingWordsLoop,lKFsSharingWordsMerge;

            // Search all keyframes that share a word with current keyframes
            // Discard keyframes connected to the query keyframe
            {
                unique_lock<mutex> lock(mMutex);

                for(DBoW3::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit != vend; vit++)
                {
                    list<KeyFrame*> &lKFs = mvInvertedFile[vit->first];

                    for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
                    {
                        KeyFrame* pKFi=*lit;
                        if(pKFi->GetMap()==pKF->GetMap()) // For consider a loop candidate it a candidate it must be in the same map
                        {
                            if(pKFi->mnLoopQuery!=pKF->mnId)
                            {
                                pKFi->mnLoopWords=0;
                                if(!spConnectedKeyFrames.count(pKFi))
                                {
                                    pKFi->mnLoopQuery=pKF->mnId;
                                    lKFsSharingWordsLoop.push_back(pKFi);
                                }
                            }
                            pKFi->mnLoopWords++;
                        }
                        else if(!pKFi->GetMap()->IsBad())
                        {
                            if(pKFi->mnMergeQuery!=pKF->mnId)
                            {
                                pKFi->mnMergeWords=0;
                                if(!spConnectedKeyFrames.count(pKFi))
                                {
                                    pKFi->mnMergeQuery=pKF->mnId;
                                    lKFsSharingWordsMerge.push_back(pKFi);
                                }
                            }
                            pKFi->mnMergeWords++;
                        }
                    }
                }
            }

            if(lKFsSharingWordsLoop.empty() && lKFsSharingWordsMerge.empty())
                return;

            if(!lKFsSharingWordsLoop.empty())
            {
                list<pair<float,KeyFrame*> > lScoreAndMatch;

                // Only compare against those keyframes that share enough words
                int maxCommonWords=0;
                for(list<KeyFrame*>::iterator lit=lKFsSharingWordsLoop.begin(), lend= lKFsSharingWordsLoop.end(); lit!=lend; lit++)
                {
                    if((*lit)->mnLoopWords>maxCommonWords)
                        maxCommonWords=(*lit)->mnLoopWords;
                }

                int minCommonWords = maxCommonWords*0.8f;

                int nscores=0;

                // Compute similarity score. Retain the matches whose score is higher than minScore
                for(list<KeyFrame*>::iterator lit=lKFsSharingWordsLoop.begin(), lend= lKFsSharingWordsLoop.end(); lit!=lend; lit++)
                {
                    KeyFrame* pKFi = *lit;

                    if(pKFi->mnLoopWords>minCommonWords)
                    {
                        nscores++;

                        float si = mpVoc->score(pKF->mBowVec,pKFi->mBowVec);

                        pKFi->mLoopScore = si;
                        if(si>=minScore)
                            lScoreAndMatch.push_back(make_pair(si,pKFi));
                    }
                }

                if(!lScoreAndMatch.empty())
                {
                    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
                    float bestAccScore = minScore;

                    // Lets now accumulate score by covisibility
                    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
                    {
                        KeyFrame* pKFi = it->second;
                        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

                        float bestScore = it->first;
                        float accScore = it->first;
                        KeyFrame* pBestKF = pKFi;
                        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
                        {
                            KeyFrame* pKF2 = *vit;
                            if(pKF2->mnLoopQuery==pKF->mnId && pKF2->mnLoopWords>minCommonWords)
                            {
                                accScore+=pKF2->mLoopScore;
                                if(pKF2->mLoopScore>bestScore)
                                {
                                    pBestKF=pKF2;
                                    bestScore = pKF2->mLoopScore;
                                }
                            }
                        }

                        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
                        if(accScore>bestAccScore)
                            bestAccScore=accScore;
                    }

                    // Return all those keyframes with a score higher than 0.75*bestScore
                    float minScoreToRetain = 0.75f*bestAccScore;

                    set<KeyFrame*> spAlreadyAddedKF;
                    vpLoopCand.reserve(lAccScoreAndMatch.size());

                    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
                    {
                        if(it->first>minScoreToRetain)
                        {
                            KeyFrame* pKFi = it->second;
                            if(!spAlreadyAddedKF.count(pKFi))
                            {
                                vpLoopCand.push_back(pKFi);
                                spAlreadyAddedKF.insert(pKFi);
                            }
                        }
                    }
                }

            }

            if(!lKFsSharingWordsMerge.empty())
            {
                list<pair<float,KeyFrame*> > lScoreAndMatch;

                // Only compare against those keyframes that share enough words
                int maxCommonWords=0;
                for(list<KeyFrame*>::iterator lit=lKFsSharingWordsMerge.begin(), lend=lKFsSharingWordsMerge.end(); lit!=lend; lit++)
                {
                    if((*lit)->mnMergeWords>maxCommonWords)
                        maxCommonWords=(*lit)->mnMergeWords;
                }

                int minCommonWords = maxCommonWords*0.8f;

                int nscores=0;

                // Compute similarity score. Retain the matches whose score is higher than minScore
                for(list<KeyFrame*>::iterator lit=lKFsSharingWordsMerge.begin(), lend=lKFsSharingWordsMerge.end(); lit!=lend; lit++)
                {
                    KeyFrame* pKFi = *lit;

                    if(pKFi->mnMergeWords>minCommonWords)
                    {
                        nscores++;

                        float si = mpVoc->score(pKF->mBowVec,pKFi->mBowVec);

                        pKFi->mMergeScore = si;
                        if(si>=minScore)
                            lScoreAndMatch.push_back(make_pair(si,pKFi));
                    }
                }

                if(!lScoreAndMatch.empty())
                {
                    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
                    float bestAccScore = minScore;

                    // Lets now accumulate score by covisibility
                    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
                    {
                        KeyFrame* pKFi = it->second;
                        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

                        float bestScore = it->first;
                        float accScore = it->first;
                        KeyFrame* pBestKF = pKFi;
                        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
                        {
                            KeyFrame* pKF2 = *vit;
                            if(pKF2->mnMergeQuery==pKF->mnId && pKF2->mnMergeWords>minCommonWords)
                            {
                                accScore+=pKF2->mMergeScore;
                                if(pKF2->mMergeScore>bestScore)
                                {
                                    pBestKF=pKF2;
                                    bestScore = pKF2->mMergeScore;
                                }
                            }
                        }

                        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
                        if(accScore>bestAccScore)
                            bestAccScore=accScore;
                    }

                    // Return all those keyframes with a score higher than 0.75*bestScore
                    float minScoreToRetain = 0.75f*bestAccScore;

                    set<KeyFrame*> spAlreadyAddedKF;
                    vpMergeCand.reserve(lAccScoreAndMatch.size());

                    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
                    {
                        if(it->first>minScoreToRetain)
                        {
                            KeyFrame* pKFi = it->second;
                            if(!spAlreadyAddedKF.count(pKFi))
                            {
                                vpMergeCand.push_back(pKFi);
                                spAlreadyAddedKF.insert(pKFi);
                            }
                        }
                    }
                }

            }

            for(DBoW3::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit != vend; vit++)
            {
                list<KeyFrame*> &lKFs = mvInvertedFile[vit->first];

                for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
                {
                    KeyFrame* pKFi=*lit;
                    pKFi->mnLoopQuery=-1;
                    pKFi->mnMergeQuery=-1;
                }
            }

        }

        void KeyFrameDatabase::DetectBestCandidates(KeyFrame *pKF, vector<KeyFrame*> &vpLoopCand, vector<KeyFrame*> &vpMergeCand, int nMinWords)
        {
            list<KeyFrame*> lKFsSharingWords;
            set<KeyFrame*> spConnectedKF;

            // Search all keyframes that share a word with current frame
            {
                unique_lock<mutex> lock(mMutex);

                spConnectedKF = pKF->GetConnectedKeyFrames();

                for(DBoW3::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit != vend; vit++)
                {
                    list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

                    for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
                    {
                        KeyFrame* pKFi=*lit;
                        if(spConnectedKF.find(pKFi) != spConnectedKF.end())
                        {
                            continue;
                        }
                        if(pKFi->mnPlaceRecognitionQuery!=pKF->mnId)
                        {
                            pKFi->mnPlaceRecognitionWords=0;
                            pKFi->mnPlaceRecognitionQuery=pKF->mnId;
                            lKFsSharingWords.push_back(pKFi);
                        }
                        pKFi->mnPlaceRecognitionWords++;

                    }
                }
            }
            if(lKFsSharingWords.empty())
                return;

            // Only compare against those keyframes that share enough words
            int maxCommonWords=0;
            for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
            {
                if((*lit)->mnPlaceRecognitionWords>maxCommonWords)
                    maxCommonWords=(*lit)->mnPlaceRecognitionWords;
            }

            int minCommonWords = maxCommonWords*0.8f;

            if(minCommonWords < nMinWords)
            {
                minCommonWords = nMinWords;
            }

            list<pair<float,KeyFrame*> > lScoreAndMatch;

            int nscores=0;

            // Compute similarity score.
            for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi = *lit;

                if(pKFi->mnPlaceRecognitionWords>minCommonWords)
                {
                    nscores++;
                    float si = mpVoc->score(pKF->mBowVec,pKFi->mBowVec);
                    pKFi->mPlaceRecognitionScore=si;
                    lScoreAndMatch.push_back(make_pair(si,pKFi));
                }
            }

            if(lScoreAndMatch.empty())
                return;

            list<pair<float,KeyFrame*> > lAccScoreAndMatch;
            float bestAccScore = 0;

            // Lets now accumulate score by covisibility
            for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
            {
                KeyFrame* pKFi = it->second;
                vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

                float bestScore = it->first;
                float accScore = bestScore;
                KeyFrame* pBestKF = pKFi;
                for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
                {
                    KeyFrame* pKF2 = *vit;
                    if(pKF2->mnPlaceRecognitionQuery!=pKF->mnId)
                        continue;

                    accScore+=pKF2->mPlaceRecognitionScore;
                    if(pKF2->mPlaceRecognitionScore>bestScore)
                    {
                        pBestKF=pKF2;
                        bestScore = pKF2->mPlaceRecognitionScore;
                    }

                }
                lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
                if(accScore>bestAccScore)
                    bestAccScore=accScore;
            }

            // Return all those keyframes with a score higher than 0.75*bestScore
            float minScoreToRetain = 0.75f*bestAccScore;
            set<KeyFrame*> spAlreadyAddedKF;
            vpLoopCand.reserve(lAccScoreAndMatch.size());
            vpMergeCand.reserve(lAccScoreAndMatch.size());
            for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
            {
                const float &si = it->first;
                if(si>minScoreToRetain)
                {
                    KeyFrame* pKFi = it->second;
                    if(!spAlreadyAddedKF.count(pKFi))
                    {
                        if(pKF->GetMap() == pKFi->GetMap())
                        {
                            vpLoopCand.push_back(pKFi);
                        }
                        else
                        {
                            vpMergeCand.push_back(pKFi);
                        }
                        spAlreadyAddedKF.insert(pKFi);
                    }
                }
            }
        }

        bool compFirst(const pair<float, KeyFrame*> & a, const pair<float, KeyFrame*> & b)
        {
            return a.first > b.first;
        }


        void KeyFrameDatabase::DetectNBestCandidates(KeyFrame *pKF, vector<KeyFrame*> &vpLoopCand, vector<KeyFrame*> &vpMergeCand, int nNumCandidates)
        {
            list<KeyFrame*> lKFsSharingWords;
            set<KeyFrame*> spConnectedKF;

            // Search all keyframes that share a word with current frame
            {
                unique_lock<mutex> lock(mMutex);

                spConnectedKF = pKF->GetConnectedKeyFrames();

                for(DBoW3::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit != vend; vit++)
                {
                    list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

                    for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
                    {
                        KeyFrame* pKFi=*lit;
                        if(pKFi->mnPlaceRecognitionQuery!=pKF->mnId)
                        {
                            pKFi->mnPlaceRecognitionWords=0;
                            if(!spConnectedKF.count(pKFi))
                            {

                                pKFi->mnPlaceRecognitionQuery=pKF->mnId;
                                lKFsSharingWords.push_back(pKFi);
                            }
                        }
                        pKFi->mnPlaceRecognitionWords++;

                    }
                }
            }
            if(lKFsSharingWords.empty())
                return;

            // Only compare against those keyframes that share enough words
            int maxCommonWords=0;
            for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
            {
                if((*lit)->mnPlaceRecognitionWords>maxCommonWords)
                    maxCommonWords=(*lit)->mnPlaceRecognitionWords;
            }

            int minCommonWords = maxCommonWords*0.8f;

            list<pair<float,KeyFrame*> > lScoreAndMatch;

            int nscores=0;

            // Compute similarity score.
            for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi = *lit;

                if(pKFi->mnPlaceRecognitionWords>minCommonWords)
                {
                    nscores++;
                    float si = mpVoc->score(pKF->mBowVec,pKFi->mBowVec);
                    pKFi->mPlaceRecognitionScore=si;
                    lScoreAndMatch.push_back(make_pair(si,pKFi));
                }
            }

            if(lScoreAndMatch.empty())
                return;

            list<pair<float,KeyFrame*> > lAccScoreAndMatch;
            float bestAccScore = 0;

            // Lets now accumulate score by covisibility
            for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
            {
                KeyFrame* pKFi = it->second;
                vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

                float bestScore = it->first;
                float accScore = bestScore;
                KeyFrame* pBestKF = pKFi;
                for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
                {
                    KeyFrame* pKF2 = *vit;
                    if(pKF2->mnPlaceRecognitionQuery!=pKF->mnId)
                        continue;

                    accScore+=pKF2->mPlaceRecognitionScore;
                    if(pKF2->mPlaceRecognitionScore>bestScore)
                    {
                        pBestKF=pKF2;
                        bestScore = pKF2->mPlaceRecognitionScore;
                    }

                }
                lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
                if(accScore>bestAccScore)
                    bestAccScore=accScore;
            }

            lAccScoreAndMatch.sort(compFirst);

            vpLoopCand.reserve(nNumCandidates);
            vpMergeCand.reserve(nNumCandidates);
            set<KeyFrame*> spAlreadyAddedKF;
            int i = 0;
            list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin();
            while(i < lAccScoreAndMatch.size() && (vpLoopCand.size() < nNumCandidates || vpMergeCand.size() < nNumCandidates))
            {
                KeyFrame* pKFi = it->second;
                if(pKFi->isBad())
                    continue;

                if(!spAlreadyAddedKF.count(pKFi))
                {
                    if(pKF->GetMap() == pKFi->GetMap() && vpLoopCand.size() < nNumCandidates)
                    {
                        vpLoopCand.push_back(pKFi);
                    }
                    else if(pKF->GetMap() != pKFi->GetMap() && vpMergeCand.size() < nNumCandidates && !pKFi->GetMap()->IsBad())
                    {
                        vpMergeCand.push_back(pKFi);
                    }
                    spAlreadyAddedKF.insert(pKFi);
                }
                i++;
                it++;
            }
        }

        vector<KeyFrame*> KeyFrameDatabase::DetectRelocalizationCandidates(Frame *F, Map* pMap) {
            list<KeyFrame *> lKFsSharingWords;

            // Search all keyframes that share a word with current frame
            {
                unique_lock<mutex> lock(mMutex);

                for (DBoW3::BowVector::const_iterator vit = F->mBowVec.begin(), vend = F->mBowVec.end();
                     vit != vend; vit++) {
                    list<KeyFrame *> &lKFs = mvInvertedFile[vit->first];

                    for (list<KeyFrame *>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++) {
                        KeyFrame *pKFi = *lit;
                        if (pKFi->mnRelocQuery != F->id_) {
                            pKFi->mnRelocWords = 0;
                            pKFi->mnRelocQuery = F->id_;
                            lKFsSharingWords.push_back(pKFi);
                        }
                        pKFi->mnRelocWords++;
                    }
                }
            }
            if (lKFsSharingWords.empty())
                return vector<KeyFrame *>();

            // Only compare against those keyframes that share enough words
            int maxCommonWords = 0;
            for (list<KeyFrame *>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end();
                 lit != lend; lit++) {
                if ((*lit)->mnRelocWords > maxCommonWords)
                    maxCommonWords = (*lit)->mnRelocWords;
            }

            int minCommonWords = maxCommonWords * 0.8f;

            list<pair<float, KeyFrame *> > lScoreAndMatch;

            int nscores = 0;

            // Compute similarity score.
            for (list<KeyFrame *>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end();
                 lit != lend; lit++) {
                KeyFrame *pKFi = *lit;

                if (pKFi->mnRelocWords > minCommonWords) {
                    nscores++;
                    float si = mpVoc->score(F->mBowVec, pKFi->mBowVec);
                    pKFi->mRelocScore = si;
                    lScoreAndMatch.push_back(make_pair(si, pKFi));
                }
            }

            if (lScoreAndMatch.empty())
                return vector<KeyFrame *>();

            list<pair<float, KeyFrame *> > lAccScoreAndMatch;
            float bestAccScore = 0;

            // Lets now accumulate score by covisibility
            for (list<pair<float, KeyFrame *> >::iterator it = lScoreAndMatch.begin(), itend = lScoreAndMatch.end();
                 it != itend; it++) {
                KeyFrame *pKFi = it->second;
                vector<KeyFrame *> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

                float bestScore = it->first;
                float accScore = bestScore;
                KeyFrame *pBestKF = pKFi;
                for (vector<KeyFrame *>::iterator vit = vpNeighs.begin(), vend = vpNeighs.end(); vit != vend; vit++) {
                    KeyFrame *pKF2 = *vit;
                    if (pKF2->mnRelocQuery != F->id_)
                        continue;

                    accScore += pKF2->mRelocScore;
                    if (pKF2->mRelocScore > bestScore) {
                        pBestKF = pKF2;
                        bestScore = pKF2->mRelocScore;
                    }

                }
                lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
                if (accScore > bestAccScore)
                    bestAccScore = accScore;
            }

            // Return all those keyframes with a score higher than 0.75*bestScore
            float minScoreToRetain = 0.75f * bestAccScore;
            set<KeyFrame *> spAlreadyAddedKF;
            vector<KeyFrame *> vpRelocCandidates;
            vpRelocCandidates.reserve(lAccScoreAndMatch.size());
            for (list<pair<float, KeyFrame *> >::iterator it = lAccScoreAndMatch.begin(), itend = lAccScoreAndMatch.end();
                 it != itend; it++) {
                const float &si = it->first;
                if (si > minScoreToRetain) {
                    KeyFrame *pKFi = it->second;
                    if (pKFi->GetMap() != pMap)
                        continue;
                    if (!spAlreadyAddedKF.count(pKFi)) {
                        vpRelocCandidates.push_back(pKFi);
                        spAlreadyAddedKF.insert(pKFi);
                    }
                }
            }

            return vpRelocCandidates;
        }
        void KeyFrameDatabase::SetORBVocabulary(DBoW3::Vocabulary* pORBVoc)
        {
            DBoW3::Vocabulary** ptr;
            ptr = (DBoW3::Vocabulary**)( &mpVoc );
            *ptr = pORBVoc;

            mvInvertedFile.clear();
            mvInvertedFile.resize(mpVoc->size());
        }
    }
}

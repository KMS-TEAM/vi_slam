//
// Created by lacie on 11/06/2021.
//

#include "vi_slam/core/monoinitializer.h"
#include "vi_slam/geometry/epipolar_geometry.h"
#include "vi_slam/geometry/motion_estimation.h"
#include "DBoW3/DUtils/Random.h"

#include <mutex>
#include <thread>

using namespace std;

namespace vi_slam{
    namespace core{

        MonoInitializer::MonoInitializer(const datastructures::Frame &ReferenceFrame, float sigma, int iterations)
        {
            mK = ReferenceFrame.mK.clone();

            mvKeys1 = ReferenceFrame.ukeypoints_;

            mSigma = sigma;
            mSigma2 = sigma*sigma;
            mMaxIterations = iterations;
        }

        bool MonoInitializer::Initialize(const datastructures::Frame &CurrentFrame, const vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21,
                                     vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated)
        {
            // Fill structures with current keypoints and matches with reference frame
            // Reference Frame: 1, Current Frame: 2
            mvKeys2 = CurrentFrame.ukeypoints_;

            mvMatches12.clear();
            mvMatches12.reserve(mvKeys2.size());
            mvbMatched1.resize(mvKeys1.size());
            for(size_t i=0, iend=vMatches12.size();i<iend; i++)
            {
                if(vMatches12[i]>=0)
                {
                    mvMatches12.push_back(make_pair(i,vMatches12[i]));
                    mvbMatched1[i]=true;
                }
                else
                    mvbMatched1[i]=false;
            }

            const int N = mvMatches12.size();

            // Indices for minimum set selection
            vector<size_t> vAllIndices;
            vAllIndices.reserve(N);
            vector<size_t> vAvailableIndices;

            for(int i=0; i<N; i++)
            {
                vAllIndices.push_back(i);
            }

            // Generate sets of 8 points for each RANSAC iteration
            mvSets = vector< vector<size_t> >(mMaxIterations,vector<size_t>(8,0));

            DUtils::Random::SeedRandOnce(0);

            for(int it=0; it<mMaxIterations; it++)
            {
                vAvailableIndices = vAllIndices;

                // Select a minimum set
                for(size_t j=0; j<8; j++)
                {
                    int randi = DUtils::Random::RandomInt(0,vAvailableIndices.size()-1);
                    int idx = vAvailableIndices[randi];

                    mvSets[it][j] = idx;

                    vAvailableIndices[randi] = vAvailableIndices.back();
                    vAvailableIndices.pop_back();
                }
            }

            // Launch threads to compute in parallel a fundamental matrix and a homography
            vector<bool> vbMatchesInliersH, vbMatchesInliersF;
            float SH, SF;
            cv::Mat H, F;

            std::thread threadH(&geometry::FindHomography, ref(mvKeys1), ref(mvKeys2),
                                                           ref(vbMatchesInliersH),
                                                           ref(mvMatches12),
                                                           ref(SH), ref(H),
                                                           ref(mMaxIterations), ref(mSigma), ref(mvSets));

            std::thread threadF(&geometry::FindFundamental, ref(mvKeys1), ref(mvKeys2),
                                ref(vbMatchesInliersF),
                                ref(mvMatches12),
                                ref(mMaxIterations), ref(mSigma),
                                ref(SF), ref(F), ref(mvSets));

            // Wait until both threads have finished
            threadH.join();
            threadF.join();

            // Compute ratio of scores
            float RH = SH/(SH+SF);

            // Try to reconstruct from homography or fundamental depending on the ratio (0.40-0.45)
            if(RH>0.40)
                return geometry::ReconstructH(vbMatchesInliersH,mvMatches12, mvKeys1, mvKeys2, H,mK,R21,t21,vP3D,vbTriangulated,1.0, mSigma,50);
            else //if(pF_HF>0.6)
                return geometry::ReconstructF(vbMatchesInliersF,mvMatches12, mvKeys1, mvKeys2, F,mK,R21,t21,vP3D,vbTriangulated,1.0, mSigma, 50);

            return false;
        }
    }
}

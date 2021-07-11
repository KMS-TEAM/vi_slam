//
// Created by cit-industry on 11/07/2021.
//

#ifndef VI_SLAM_ISAM2OPTIMIZER_H
#define VI_SLAM_ISAM2OPTIMIZER_H

#include "vi_slam/common_include.h"

// ISAM2 INCLUDES
/* ************************************************************************* */

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>

// Each variable in the system (poses and landmarks) must be identified with a unique key.
// We can either use simple integer keys (1, 2, 3, ...) or symbols (X1, X2, L1).
// Here we will use Symbols
#include <gtsam/inference/Symbol.h>

// We want to use iSAM2 to solve the structure-from-motion problem incrementally, so
// include iSAM2 here
#include <gtsam/nonlinear/ISAM2.h>

// iSAM2 requires as input a set of new factors to be added stored in a factor graph,
// and initial guesses for any new variables used in the added factors
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

// In GTSAM, measurement functions are represented as 'factors'. Several common factors
// have been provided with the library for solving robotics/SLAM/Bundle Adjustment problems.
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/StereoFactor.h>
#include <gtsam/navigation/ImuFactor.h> // **
#include <gtsam/navigation/CombinedImuFactor.h> // **
#include <gtsam/slam/BetweenFactor.h> // **

// ADDITIONAL INCLUDES
/* ************************************************************************* */

#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

#include "vi_slam/datastructures/keyframe.h"
#include "vi_slam/datastructures/mappoint.h"
#include "vi_slam/datastructures/imu.h"
#include "vi_slam/basics/yaml.h"

#include "vi_slam/core/tracking.h"

namespace vi_slam{

    namespace basics{
        class Yaml;
    }

    namespace datastructures{
        class KeyFrame;
        class MapPoint;

        namespace IMU{
            class Point;
            class Preintegrated;
            class Calib;
            class Bias;
        }
    }

    namespace core{
        class Traking;
    }
    
    namespace optimization{

        using namespace datastructures;
        using namespace core;
        using namespace std;
        using namespace gtsam;

        class iSAM2Optimizer {

        private:
            int pose_id;
            // Create iSAM2 object
            unique_ptr<ISAM2> isam;

            // Initialize Factor Graph and Values Estimates on Nodes (continually updated by isam.update())
            NonlinearFactorGraph graph;
            Values newNodes;
            Values optimizedNodes; // current estimate of values
            Pose3 prev_camera_pose;       // current estimate of previous pose
            Pose3 prev_camera_odom;       // previous pose outputted by camera
            imuBias::ConstantBias prev_robot_bias; // **

            // Initialize IMU Variables // **
            PreintegratedCombinedMeasurements* imu_preintegrated; // CHANGE BACK TO COMBINED (Combined<->Imu)
            double prev_imu_timestamp;

            // Initialize VIO Variables
            double f;                     // Camera calibration intrinsics
            double cx;
            double cy;
            double resolution_x;          // Image distortion intrinsics
            double resolution_y;
            Cal3_S2Stereo::shared_ptr K;  // Camera calibration intrinsic matrix
            double Tx;                    // Camera calibration extrinsic: distance from cam0 to cam1
            gtsam::Matrix4 T_cam_imu_mat; // Transform to get to camera IMU frame from camera frame

            // Noise models
            noiseModel::Diagonal::shared_ptr pose_noise = noiseModel::Diagonal::Sigmas(
                    (Vector(6) << Vector3::Constant(0.5),Vector3::Constant(0.1)).finished()
            ); // (roll,pitch,yaw in rad; std on x,y,z in meters)
            noiseModel::Diagonal::shared_ptr velocity_noise = noiseModel::Isotropic::Sigma(3, 0.1); // (dim, sigma in m/s)
            noiseModel::Diagonal::shared_ptr bias_noise = noiseModel::Isotropic::Sigma(6, 1e-3); // (dim, sigma)
            noiseModel::Isotropic::shared_ptr pose_landmark_noise = noiseModel::Isotropic::Sigma(3, 1.0); // (dim, sigma in pixels): one pixel in u and v
            noiseModel::Isotropic::shared_ptr landmark_noise = noiseModel::Isotropic::Sigma(3, 0.1);

            // basics::Yaml config;

            // Imu preintegration from last frame
            IMU::Preintegrated *mpImuPreintegratedFromLastKF;

        public:
            iSAM2Optimizer(std::string& config_path);
            ~iSAM2Optimizer();

            void initializeIMUParameters(Tracking& tracker);

            void optimizerIMU();
            void optimizerVIO_IMU();
            void optimizerVIO();
        };
    }
}

#endif //VI_SLAM_ISAM2OPTIMIZER_H

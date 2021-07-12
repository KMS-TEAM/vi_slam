//
// Created by lacie on 09/07/2021.
//

#ifndef VI_SLAM_GTSAMOPTIMIZER_H
#define VI_SLAM_GTSAMOPTIMIZER_H

#include "vi_slam/common_include.h"
#include "vi_slam/datastructures/imu.h"
#include "vi_slam/datastructures/keyframe.h"
#include "vi_slam/basics/yaml.h"

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

#include <mutex>
#include <set>
#include <algorithm>
#include <queue>

#include "vi_slam/common_include.h"
#include "vi_slam/basics/converter.h"
#include "vi_slam/datastructures/frame.h"
#include "vi_slam/datastructures/keyframe.h"
#include "vi_slam/datastructures/mappoint.h"

// For keys representation
#include <gtsam/inference/Symbol.h>

// For keyframes pose
#include <gtsam/geometry/StereoCamera.h>

// For between factors
#include <gtsam/slam/BetweenFactor.h>

// For landmarks position
#include <gtsam/geometry/Point3.h>

// For first keyframe pose
//#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/slam/PriorFactor.h>

//// Mono ////
// For landmarks coordinates
#include <gtsam/geometry/Point2.h>
// For factors between keyframe and landmarks
#include <gtsam/slam/ProjectionFactor.h>

//// Stereo ////
// For landmarks coordinates
#include <gtsam/geometry/StereoPoint2.h>
// For factors between keyframe and landmarks
#include <gtsam/slam/StereoFactor.h>

// Factor Container
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
// Values Container
#include <gtsam/nonlinear/Values.h>

// Serialization
#include <gtsam/base/serialization.h>

// logger
#include <spdlog/spdlog.h>
#include <spdlog/sinks/rotating_file_sink.h>

namespace vi_slam {

    namespace datastructures{
        class KeyFrame;
        class MapPoint;
        class KeyFrame;
    }
    namespace optimization{

        using namespace datastructures;

        class GTSAMOptimizer {
            friend class Optimizer;

            enum FactorType {
                PRIOR,
                BETWEEN,
                MONO,
                STEREO,
                IMUFACTOR
            };
        public:
            // GTSAM Transformer Update Type
            enum UpdateType{
                BATCH=0,
                INCREMENTAL=1
            };

        public:
            GTSAMOptimizer(std::string& config_path);

            /**
             * Returns tuple contains:
             * 1. Boolean indicates if there is a new data or not
             * 2. Optional Boolean indicates if the data is incremental update
             * 3. Optional string contains graph of the added factors since the last call to the function (serialized)
             * 4. Optional vector contains the indices of the factors had removed since the last call to the function
             * 5. Optional GTSAM KeyVector contains the keys of the added states (keyframes/landmarks) since the last call to the function
             * 6. Optional GTSAM KeyVector contains the keys of the removed states (keyframes/landmarks) since the last call to the function
             * 7. Optional GTSAM Values object contains the values of entire graph (serialized)
             * 8. Optional tuple of the most recent keyframe symbol (serialized), its timestamp, and its Pose3 (serialized)
             */
            std::tuple<bool,
                    boost::optional<bool>,
                    boost::optional<std::string>,
                    boost::optional<std::vector<size_t>>,
                    boost::optional<const gtsam::KeyVector>,
                    boost::optional<const gtsam::KeyVector>,
                    boost::optional<gtsam::Values>,
                    boost::optional<std::tuple<std::string, double, std::string>>> checkForNewData();

            void setUpdateType(const UpdateType update_type);

        protected:
            void transformGraphToGtsam(const std::vector<vi_slam::datastructures::KeyFrame *> &vpKFs,
                                       const std::vector<vi_slam::datastructures::MapPoint *> &vpMP);

        private:
            bool start();
            void finish();

            void initializeIMUParameters(const IMU::Point& imu);

            void updateKeyFrame(vi_slam::datastructures::KeyFrame *pKF, bool add_between_factor = false);

            void updateLandmark(vi_slam::datastructures::MapPoint *pMP);

            void updateObservations(vi_slam::datastructures::MapPoint *pMP,
                                    const std::map<KeyFrame*,std::tuple<int,int>> &observations);

            void addMonoMeasurement(vi_slam::datastructures::KeyFrame *pKF,
                                    vi_slam::datastructures::MapPoint *pMP,
                                    Eigen::Matrix<double, 2, 1> &obs,
                                    const float inv_sigma_2);

            void addStereoMeasurement(vi_slam::datastructures::KeyFrame *pKF,
                                      vi_slam::datastructures::MapPoint *pMP,
                                      Eigen::Matrix<double, 3, 1> &obs,
                                      const float inv_sigma_2);

            void calculateDiffrencesBetweenValueSets();

            void calculateDiffrencesBetweenFactorSets();

            void exportKeysFromMap(std::map<std::pair<gtsam::Key, gtsam::Key>,
                    std::pair<std::string, FactorType>> &map,
                                   std::vector<std::pair<gtsam::Key, gtsam::Key>> &output);

            void exportValuesFromMap(std::map<std::pair<gtsam::Key, gtsam::Key>,
                    std::pair<std::string, FactorType>> &map,
                                     std::vector<std::pair<std::string, FactorType>> &output);

            std::string setToString(const std::set<gtsam::Key> &set) const;

            void createFactorGraph(std::vector<std::pair<std::string, FactorType>> ser_factors_vec, bool is_incremental);

            void createFactorGraph(std::map<std::pair<gtsam::Key, gtsam::Key>, std::pair<std::string, FactorType>> ser_factors_map,
                                                          bool is_incremental);

            std::vector<size_t> createDeletedFactorsIndicesVec(std::vector<std::pair<gtsam::Key, gtsam::Key>> &del_factors);

            // Private implementation of std::set_difference
            std::map<std::pair<gtsam::Key, gtsam::Key>, std::pair<std::string, FactorType>> getDifferenceSet(std::map<std::pair<gtsam::Key, gtsam::Key>, std::pair<std::string,FactorType>> &set_A,
                                                                                                             std::map<std::pair<gtsam::Key, gtsam::Key>, std::pair<std::string, FactorType>> &set_B);

            gtsam::KeyVector getDifferenceKeyList(const gtsam::KeyVector &vec_A, const gtsam::KeyVector &vec_B);

        public:
            // Create iSAM2 object
            unique_ptr<gtsam::ISAM2> isam;

            // Initialize Factor Graph and Values Estimates on Nodes (continually updated by isam.update())
            gtsam::NonlinearFactorGraph graph;
            gtsam::Values newNodes;
            gtsam::Values optimizedNodes; // current estimate of values
            gtsam::Pose3 prev_pose;       // current estimate of previous pose
            gtsam::Vector3 prev_velocity;       // previous pose outputted by camera
            gtsam::imuBias::ConstantBias prev_imu_bias; // **

            // Initialize IMU Variables // **
            gtsam::PreintegratedCombinedMeasurements* imu_preintegrated; // CHANGE BACK TO COMBINED (Combined<->Imu)
            double prev_imu_timestamp;

            // Initialize VIO Variables
            double f;                     // Camera calibration intrinsics
            double cx;
            double cy;
            double resolution_x;          // Image distortion intrinsics
            double resolution_y;
            gtsam::Cal3_S2Stereo::shared_ptr K;  // Camera calibration intrinsic matrix
            double Tx;                    // Camera calibration extrinsic: distance from cam0 to cam1
            gtsam::Matrix4 T_cam_imu_mat; // Transform to get to camera IMU frame from camera frame

            std::queue<std::tuple<bool,
                    bool,
                    std::string,
                    std::vector<size_t>,
                    const gtsam::KeyVector,
                    const gtsam::KeyVector,
                    gtsam::Values,
                    std::tuple<std::string, double, std::string>>> ready_data_queue_;

            gtsam::Values session_values_, last_session_values_;
            gtsam::Cal3_S2Stereo::shared_ptr cam_params_stereo_;
            gtsam::Cal3_S2::shared_ptr cam_params_mono_;

            gtsam::noiseModel::Diagonal::shared_ptr between_factors_prior_;
            gtsam::noiseModel::Diagonal::shared_ptr pose_noise;
            gtsam::noiseModel::Diagonal::shared_ptr velocity_noise;
            gtsam::noiseModel::Diagonal::shared_ptr bias_noise;
            gtsam::noiseModel::Isotropic::shared_ptr pose_landmark_noise;
            gtsam::noiseModel::Isotropic::shared_ptr landmark_noise;

            bool is_cam_params_initialized_ = false;
            std::vector<std::pair<std::string, FactorType>> add_factors_;
            std::vector<std::pair<gtsam::Key, gtsam::Key>> del_factors_;
            std::map<std::pair<gtsam::Key, gtsam::Key>, std::pair<std::string, FactorType>> session_factors_, last_session_factors_;
            gtsam::KeyVector del_states_, add_states_;
            std::tuple<std::string, double, std::string> recent_kf_;
            std::map<std::pair<gtsam::Key, gtsam::Key>, size_t> factor_indecies_dict_;
            size_t current_index_ = 0;
            UpdateType update_type_;

            std::mutex mutex_;

            std::shared_ptr<spdlog::logger> logger_;
        };
    }
}

#endif //VI_SLAM_ISAM2OPTIMIZER_H

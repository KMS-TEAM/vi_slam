//
// Created by lacie on 09/07/2021.
//

#include "vi_slam/optimization/gtsamoptimizer.h"

#include "vi_slam/datastructures/keyframe.h"
#include "vi_slam/datastructures/mappoint.h"

#include "vi_slam/optimization/optimizer.h"
#include "vi_slam/optimization/gtsamoptimizer.h"
#include "vi_slam/optimization/gtsamserialization.h"
#include "vi_slam/basics/yaml.h"

#include <gtsam/navigation/ImuFactor.h> // **
#include <gtsam/navigation/CombinedImuFactor.h> // **
#include <gtsam/slam/BetweenFactor.h> // **

//#define DEBUG

namespace vi_slam {

    namespace basics{
        class Yaml;
    }

    namespace optimization{

        GTSAMOptimizer::GTSAMOptimizer(std::string& config_path) {

            basics::Yaml readConfig(config_path);

            // YAML intrinsics (pinhole): [fu fv pu pv]
            std::vector<double> cam0_intrinsics(4);
            cam0_intrinsics = readConfig.get_vec<double>("cam0/resolution");
            this->f = (cam0_intrinsics[0] + cam0_intrinsics[1]) / 2;
            this->cx = cam0_intrinsics[2];
            this->cy = cam0_intrinsics[3];

            // YAML image resolution parameters (radtan): [k1 k2 r1 r2]
            vector<double> cam0_resolution(2);
            cam0_resolution = readConfig.get_vec<double>("cam0/resolution");
            this->resolution_x = cam0_resolution[0];
            this->resolution_y = cam0_resolution[1];

            // YAML extrinsics (distance between 2 cameras and transform between imu and camera))
            vector<double> T_cam1(16);
            T_cam1 = readConfig.get_vec<double>("cam1/T_cn_cnm1");
            this->Tx = T_cam1[3];

            vector<double> T_cam_imu(16);
            T_cam_imu = readConfig.get_vec<double>("cam0/T_cam_imu");
            gtsam::Matrix4 T_cam_imu_mat_copy(T_cam_imu.data());
            T_cam_imu_mat = move(T_cam_imu_mat_copy);

            // Set K: (fx, fy, s, u0, v0, b) (b: baseline where Z = f*d/b; Tx is negative)
            this->K.reset(new Cal3_S2Stereo(cam0_intrinsics[0], cam0_intrinsics[1], 0.0,
                                            this->cx, this->cy, -this->Tx));

            // iSAM2 settings
            ISAM2Params parameters;
            parameters.relinearizeThreshold = 0.01;
            parameters.relinearizeSkip = 1;
            isam.reset(new ISAM2(parameters));

            logger_ = spdlog::rotating_logger_st("GtsamTransformer",
                                                 "GtsamTransformer.log",
                                                 1048576 * 50,
                                                 3);
#ifdef DEBUG
            logger_->set_level(spdlog::level::debug);
#else
            logger_->set_level(spdlog::level::info);
#endif
            logger_->info("CTOR - GtsamTransformer instance created");
            between_factors_prior_ = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e2, 1e2, 1e2, 1, 1, 1).finished());
            // Noise models
            pose_noise = noiseModel::Diagonal::Sigmas(
                    (Vector(6) << Vector3::Constant(0.5),Vector3::Constant(0.1)).finished()); // (roll,pitch,yaw in rad; std on x,y,z in meters)
            velocity_noise = noiseModel::Isotropic::Sigma(3, 0.1); // (dim, sigma in m/s)
            bias_noise = noiseModel::Isotropic::Sigma(6, 1e-3); // (dim, sigma)
            pose_landmark_noise = noiseModel::Isotropic::Sigma(3, 1.0); // (dim, sigma in pixels): one pixel in u and v
            landmark_noise = noiseModel::Isotropic::Sigma(3, 0.1);
        }

        void GTSAMOptimizer::initializeIMUParameters(const IMU::Point &imu) {

            // Get (constant) IMU covariance of angular vel, and linear acc (row major about x, y, z axes)
            boost::array<double, 9> ang_vel_cov;
            boost::array<double, 9> lin_acc_cov;

            lin_acc_cov[0] = 0.04;
            lin_acc_cov[1] = 0;
            lin_acc_cov[2] = 0;

            lin_acc_cov[3] = 0;
            lin_acc_cov[4] = 0.04;
            lin_acc_cov[5] = 0;

            lin_acc_cov[6] = 0;
            lin_acc_cov[7] = 0;
            lin_acc_cov[8] = 0.04;

            ang_vel_cov[0] = 0.02;
            ang_vel_cov[1] = 0;
            ang_vel_cov[2] = 0;

            ang_vel_cov[3] = 0;
            ang_vel_cov[4] = 0.02;
            ang_vel_cov[5] = 0;

            ang_vel_cov[6] = 0;
            ang_vel_cov[7] = 0;
            ang_vel_cov[8] = 0.02;

            // Convert covariances to matrix form (Eigen::Matrix<float, 3, 3>)
            // gtsam::Matrix3 orient_cov_mat(orient_cov.data());
            gtsam::Matrix3 ang_vel_cov_mat(ang_vel_cov.data());
            gtsam::Matrix3 lin_acc_cov_mat(lin_acc_cov.data());
            // std::cout << "Orientation Covariance Matrix (not used): " << std::endl << orient_cov_mat << std::endl;
            std::cout << "Angular Velocity Covariance Matrix: " << std::endl << ang_vel_cov_mat << std::endl;
            std::cout << "Linear Acceleration Covariance Matrix: " << std::endl << lin_acc_cov_mat << std::endl;

            // Assign IMU preintegration parameters
            boost::shared_ptr<PreintegratedCombinedMeasurements::Params> p = PreintegratedCombinedMeasurements::Params::MakeSharedU();
            p->n_gravity = gtsam::Vector3(-imu.a.x, -imu.a.y, -imu.a.z);
            p->accelerometerCovariance = lin_acc_cov_mat;
            p->integrationCovariance = Matrix33::Identity(3,3)*1e-8; // (DON'T USE "orient_cov_mat": ALL ZEROS)
            p->gyroscopeCovariance = ang_vel_cov_mat;
            p->biasAccCovariance = Matrix33::Identity(3,3)*pow(0.004905,2);
            p->biasOmegaCovariance = Matrix33::Identity(3,3)*pow(0.000001454441043,2);
            p->biasAccOmegaInt = Matrix::Identity(6,6)*1e-5;
            imu_preintegrated = reinterpret_cast<PreintegratedCombinedMeasurements *>(new PreintegratedImuMeasurements(
                    p, gtsam::imuBias::ConstantBias())); // CHANGE BACK TO COMBINED: (Combined<->Imu)
        }

        void GTSAMOptimizer::addMonoMeasurement(vi_slam::datastructures::KeyFrame *pKF,
                                                  vi_slam::datastructures::MapPoint *pMP,
                                                  Eigen::Matrix<double, 2, 1> &obs,
                                                  const float inv_sigma_2) {
            logger_->debug("addMonoMeasurement - pKF->mnId: {}, pMP->mnId: {}", pKF->mnId, pMP->id_);
            if (!is_cam_params_initialized_) {
                std::cout << "addMonoMeasurement - camera params has not been initialized!" << std::endl;
                exit(-2);
            }
            // Create both symbols
            gtsam::Symbol keyframe_sym('x', pKF->mnId);
            gtsam::Symbol landmark_sym('l', pMP->id_);

            // Create landmark observation
            gtsam::Point2 obs_gtsam(obs(0), obs(1));

            // Create factor graph
            gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>
                    factor(obs_gtsam,
                           gtsam::noiseModel::Diagonal::Variances(Eigen::Vector2d(1 / inv_sigma_2, 1 / inv_sigma_2)),
                           keyframe_sym.key(),
                           landmark_sym.key(),
                           cam_params_mono_);
            session_factors_[std::make_pair(keyframe_sym.key(), landmark_sym.key())] = std::make_pair(gtsam::serialize(factor), FactorType::MONO);
        }

        void GTSAMOptimizer::addStereoMeasurement(vi_slam::datastructures::KeyFrame *pKF,
                                                    vi_slam::datastructures::MapPoint *pMP,
                                                    Eigen::Matrix<double, 3, 1> &obs,
                                                    const float inv_sigma_2) {
            logger_->debug("addStereoMeasurement - pKF->mnId: {}, pMP->mnId: {}", pKF->mnId, pMP->id_);
            // logger_->debug("addStereoMeasurement - pKF->mnId: {}, pMP->mnId: {}", pKF->mnId, pMP->id_);
            if (!is_cam_params_initialized_) {
                std::cout << "addStereoMeasurement - camera params has not been initialized!" << std::endl;
                exit(-2);
            }
            // Create both symbols
            gtsam::Symbol keyframe_sym('x', pKF->mnId);
            gtsam::Symbol landmark_sym('l', pMP->id_);

            // Create landmark observation
            gtsam::StereoPoint2 obs_gtsam(obs(0), obs(2), obs(1));

            // Create factor graph
            gtsam::GenericStereoFactor<gtsam::Pose3, gtsam::Point3>
                    factor(obs_gtsam,
                           gtsam::noiseModel::Diagonal::Variances(Eigen::Vector3d(1 / inv_sigma_2, 1 / inv_sigma_2, 1 / inv_sigma_2)),
                           keyframe_sym.key(),
                           landmark_sym.key(),
                           cam_params_stereo_);
            session_factors_[std::make_pair(keyframe_sym.key(), landmark_sym.key())] = std::make_pair(gtsam::serialize(factor), FactorType::STEREO);

            graph.emplace_shared<gtsam::GenericStereoFactor<gtsam::Pose3, gtsam::Point3>>(obs_gtsam,
                                                                                          gtsam::noiseModel::Diagonal::Variances(Eigen::Vector3d(1 / inv_sigma_2, 1 / inv_sigma_2, 1 / inv_sigma_2)),
                                                                                          keyframe_sym.key(),
                                                                                          landmark_sym.key(),
                                                                                          cam_params_stereo_);
        }

        std::tuple<bool,
                boost::optional<bool>,
                boost::optional<std::string>,
                boost::optional<std::vector<size_t>>,
                boost::optional<const gtsam::KeyVector>,
                boost::optional<const gtsam::KeyVector>,
                boost::optional<gtsam::Values>,
                boost::optional<std::tuple<std::string, double, std::string>>> GTSAMOptimizer::checkForNewData() {
            if (ready_data_queue_.empty()) {
                logger_->debug("checkForNewData - there is no new data.");
                return std::make_tuple(false, boost::none, boost::none, boost::none, boost::none, boost::none, boost::none, boost::none);
            }
            std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
            if (lock.owns_lock()) {
                logger_->info("checkForNewData - returning new optimized data. ready_data_queue.size: {}", ready_data_queue_.size());
                auto data = ready_data_queue_.front();
                ready_data_queue_.pop();
                std::cout << "checkForNewData - returns " << (std::get<1>(data) ? "Incremental" : "Batch") << " update" << std::endl;
                return data;
            } else {
                logger_->error("checkForNewData - can't own mutex. returning false");
                return std::make_tuple(false, boost::none, boost::none, boost::none, boost::none, boost::none, boost::none, boost::none);
            }
        }

        bool GTSAMOptimizer::start() {
            std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
            if (lock.owns_lock()) {
                logger_->info("start - new recovering session.");
                add_states_.clear();
                del_states_.clear();
                session_values_.clear();

                add_factors_.clear();
                del_factors_.clear();
                session_factors_.clear();

                // iSAM2 settings
                ISAM2Params parameters;
                parameters.relinearizeThreshold = 0.01;
                parameters.relinearizeSkip = 1;
                isam.reset(new ISAM2(parameters));

                return true;
            } else {
                logger_->warn("start - can't own mutex. returns");
            }
            return false;
        }

        void GTSAMOptimizer::finish() {
            std::unique_lock<std::mutex> *lock;

            do {
                lock = new std::unique_lock<std::mutex>(mutex_, std::try_to_lock);
            } while (!lock->owns_lock());
            logger_->info("finish - ending recovering session. new_optimized_data is now available");
            logger_->info("finish - active states set size: {}", session_values_.size());
            logger_->info("finish - active factors vector size: {}", session_factors_.size());

            isam->update(graph, newNodes);
            optimizedNodes = isam->calculateEstimate();

            graph.resize(0);
            newNodes.clear();

            if (update_type_ == INCREMENTAL) {
                // Incremental update
                gtsam::NonlinearFactorGraph incremental_factor_graph = createFactorGraph(add_factors_, true);
                ready_data_queue_.emplace(true,
                                          true,
                                          gtsam::serialize(incremental_factor_graph),
                                          createDeletedFactorsIndicesVec(del_factors_),
                                          add_states_,
                                          del_states_,
                                          session_values_,
                                          recent_kf_);
            } else if (update_type_ == BATCH) {
                // Batch update
                gtsam::NonlinearFactorGraph active_factor_graph = createFactorGraph(session_factors_, false);
                ready_data_queue_.emplace(true,
                                          false,
                                          gtsam::serialize(active_factor_graph),
                                          createDeletedFactorsIndicesVec(del_factors_),
                                          add_states_,
                                          del_states_,
                                          session_values_,
                                          recent_kf_);
                std::cerr << "Check GTSAM " << std::endl;
            }
            logger_->info("finish - ready_data_queue.size: {}", ready_data_queue_.size());

            std::cout << "finish - session_factors.size: " << session_factors_.size() << " last_session_factors.size: " << last_session_factors_.size()
                      << " add_factors.size: " << add_factors_.size()
                      << " del_factors.size: " << del_factors_.size() << " add_states.size: " << add_states_.size() << " del_states.size: "
                      << del_states_.size() << " values.size: " << session_values_.size() << " last_values.size: " << last_session_values_.size() << std::endl;

            last_session_values_ = session_values_;
            last_session_factors_ = session_factors_;

            delete lock;
        }

        void GTSAMOptimizer::exportKeysFromMap(std::map<std::pair<gtsam::Key, gtsam::Key>, std::pair<std::string, FactorType>> &map,
                                                 std::vector<std::pair<gtsam::Key, gtsam::Key>> &output) {
            for (const auto &it: map) {
                output.push_back(it.first);
            }
        }

        void GTSAMOptimizer::exportValuesFromMap(std::map<std::pair<gtsam::Key, gtsam::Key>, std::pair<std::string, FactorType>> &map,
                                                   std::vector<std::pair<std::string, FactorType>> &output) {
            for (const auto &it: map) {
                output.push_back(it.second);
            }
        }

        std::string GTSAMOptimizer::setToString(const std::set<gtsam::Key> &set) const {
            std::stringstream ss;
            for (const auto &it: set)
                ss << it << " ";
            return ss.str();
        }

        gtsam::NonlinearFactorGraph GTSAMOptimizer::createFactorGraph(std::vector<std::pair<std::string, FactorType>> ser_factors_vec,
                                                                        bool is_incremental) {
            // In use only in batch mode (not incremental)
            std::map<std::pair<gtsam::Key, gtsam::Key>, std::pair<std::string, FactorType>> new_active_factors;

            if (!is_incremental) {
                current_index_ = 0;
                factor_indecies_dict_.clear();
            }

            gtsam::NonlinearFactorGraph graph;
            for (const auto &it: ser_factors_vec) {
                switch (it.second) {
                    case FactorType::PRIOR: {
                        gtsam::PriorFactor<gtsam::Pose3> prior_factor;
                        gtsam::deserialize(it.first, prior_factor);
                        graph.push_back(prior_factor);
                        factor_indecies_dict_[std::make_pair(prior_factor.keys()[0], prior_factor.keys()[1])] = current_index_++;
                        break;
                    }
                    case FactorType::BETWEEN: {
                        gtsam::BetweenFactor<gtsam::Pose3> between_factor;
                        gtsam::deserialize(it.first, between_factor);
                        if (!is_incremental && (del_factors_.size() > 0 || del_states_.size() > 0)) {
                            gtsam::Symbol first_sym(between_factor.keys().at(0));
                            gtsam::Symbol second_sym(between_factor.keys().at(1));
                            if ((std::find(del_factors_.begin(), del_factors_.end(), std::make_pair(first_sym.key(), second_sym.key())) != del_factors_.end())
                                || (std::find(del_states_.begin(), del_states_.end(), first_sym.key()) != del_states_.end())
                                || (std::find(del_states_.begin(), del_states_.end(), second_sym.key()) != del_states_.end())) {
                                break;
                            } else {
                                new_active_factors[std::make_pair(first_sym.key(), second_sym.key())] = it;
                            }
                        }
                        graph.push_back(between_factor);
                        factor_indecies_dict_[std::make_pair(between_factor.keys()[0], between_factor.keys()[1])] = current_index_++;
                        break;
                    }
                    case FactorType::MONO: {
                        gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2> mono_factor;
                        gtsam::deserialize(it.first, mono_factor);
                        if (!is_incremental && (del_factors_.size() > 0 || del_states_.size() > 0)) {
                            gtsam::Symbol first_sym(mono_factor.keys().at(0));
                            gtsam::Symbol second_sym(mono_factor.keys().at(1));
                            if ((std::find(del_factors_.begin(), del_factors_.end(), std::make_pair(first_sym.key(), second_sym.key())) != del_factors_.end())
                                || (std::find(del_states_.begin(), del_states_.end(), first_sym.key()) != del_states_.end())
                                || (std::find(del_states_.begin(), del_states_.end(), second_sym.key()) != del_states_.end())) {
                                break;
                            } else {
                                new_active_factors[std::make_pair(first_sym.key(), second_sym.key())] = it;
                            }
                        }
                        graph.push_back(mono_factor);
                        factor_indecies_dict_[std::make_pair(mono_factor.keys()[0], mono_factor.keys()[1])] = current_index_++;
                        break;
                    }
                    case FactorType::STEREO: {
                        gtsam::GenericStereoFactor<gtsam::Pose3, gtsam::Point3> stereo_factor;
                        gtsam::deserialize(it.first, stereo_factor);
                        if (!is_incremental && (del_factors_.size() > 0 || del_states_.size() > 0)) {
                            gtsam::Symbol first_sym(stereo_factor.keys().at(0));
                            gtsam::Symbol second_sym(stereo_factor.keys().at(1));
                            if ((std::find(del_factors_.begin(), del_factors_.end(), std::make_pair(first_sym.key(), second_sym.key())) != del_factors_.end())
                                || (std::find(del_states_.begin(), del_states_.end(), first_sym.key()) != del_states_.end())
                                || (std::find(del_states_.begin(), del_states_.end(), second_sym.key()) != del_states_.end())) {
                                break;
                            } else {
                                new_active_factors[std::make_pair(first_sym.key(), second_sym.key())] = it;
                            }
                        }
                        graph.push_back(stereo_factor);
                        factor_indecies_dict_[std::make_pair(stereo_factor.keys()[0], stereo_factor.keys()[1])] = current_index_++;
                        break;
                    }
                }
            }
            std::cout << "createFactorGraph - size: " << graph.size() << std::endl;
            if (!is_incremental) {
                session_factors_ = new_active_factors;

                for (const auto &it: del_states_) {
                    if (session_values_.find(it) != session_values_.end()) {
                        session_values_.erase(it);
                    }
                }
            }

            return graph;
        }

        gtsam::NonlinearFactorGraph GTSAMOptimizer::createFactorGraph(map<pair<gtsam::Key, gtsam::Key>,
                pair<string, vi_slam::optimization::GTSAMOptimizer::FactorType>> ser_factors_map,
                                                                        bool is_incremental) {
            std::vector<std::pair<std::string, FactorType>> ser_factors_vec;
            for (const auto &it: ser_factors_map)
                ser_factors_vec.push_back(it.second);

            return createFactorGraph(ser_factors_vec, is_incremental);
        }

        std::vector<size_t> GTSAMOptimizer::createDeletedFactorsIndicesVec(std::vector<std::pair<gtsam::Key, gtsam::Key>> &del_factors) {
            std::vector<size_t> deleted_factors_indecies;
            for (const auto &it: del_factors) {
                auto dict_it = factor_indecies_dict_.find(it);
                if (dict_it != factor_indecies_dict_.end()) {
                    deleted_factors_indecies.push_back(dict_it->second);

                    gtsam::Symbol key1(it.first);
                    gtsam::Symbol key2(it.second);
                    std::cout << "createDeletedFactorsIndicesVec - " << key1.chr() << key1.index() << "-" << key2.chr() << key2.index() << " index: "
                              << dict_it->second << std::endl;
                }
            }
            return deleted_factors_indecies;
        }

        map<pair<gtsam::Key, gtsam::Key>, pair<string, GTSAMOptimizer::FactorType>> GTSAMOptimizer::getDifferenceSet(map<pair<gtsam::Key, gtsam::Key>,
                pair<string,
                        vi_slam::optimization::GTSAMOptimizer::FactorType>> &set_A,
                                                                                                                         map<pair<gtsam::Key, gtsam::Key>,
                                                                                                                                 pair<string,
                                                                                                                                         vi_slam::optimization::GTSAMOptimizer::FactorType>> &set_B) {
            map<pair<gtsam::Key, gtsam::Key>, pair<string, GTSAMOptimizer::FactorType>> diff_set;
            for (const auto &it_A: set_A) {
                if (set_B.find(it_A.first) == set_B.end()) {
                    diff_set.insert(it_A);
                }
            }
            return diff_set;
        }

        void GTSAMOptimizer::transformGraphToGtsam(const vector<vi_slam::datastructures::KeyFrame *> &vpKFs, const vector<vi_slam::datastructures::MapPoint *> &vpMP) {
            if (!start())
                return;
            for (const auto &pKF: vpKFs) {
                if (pKF->isBad())
                    continue;
                updateKeyFrame(pKF, true);
            }
            for (const auto &pMP: vpMP) {
                if (pMP->isBad())
                    continue;
                updateLandmark(pMP);
                const std::map<KeyFrame*,std::tuple<int,int>> observations = pMP->GetObservations();
                updateObservations(pMP, observations);
            }
            calculateDiffrencesBetweenValueSets();
            calculateDiffrencesBetweenFactorSets();
            finish();
        }

        void GTSAMOptimizer::updateKeyFrame(vi_slam::datastructures::KeyFrame *pKF, bool add_between_factor) {
            // Create keyframe symbol
            gtsam::Symbol sym('x', pKF->mnId);
            gtsam::Symbol v_sym('v', pKF->mnId);
            gtsam::Symbol b_sym('b', pKF->mnId);

            double dt = (pKF->mTimeStamp - prev_imu_timestamp);

            // Create camera parameters
            if (!is_cam_params_initialized_) {
                cam_params_stereo_.reset(new gtsam::Cal3_S2Stereo(pKF->fx, pKF->fy, 0.0, pKF->cx, pKF->cy, pKF->mb));
                cam_params_mono_.reset(new gtsam::Cal3_S2(cam_params_stereo_->calibration()));
                is_cam_params_initialized_ = true;
            }

            // Create pose
            cv::Mat T_cv = pKF->GetPose();
            Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor>> T_gtsam(T_cv.ptr<float>(), T_cv.rows, T_cv.cols);
            gtsam::Pose3 left_cam_pose(T_gtsam.cast<double>());
            gtsam::StereoCamera stereo_cam(left_cam_pose, cam_params_stereo_);

            session_values_.insert(sym.key(), stereo_cam.pose());

            // Adding prior factor for x0
            if (pKF->mnId == 0) {

                // GTSAMOptimizer::initializeIMUParameters(pKF->mpImuPreintegrated);

                // prev_velocity = Vector3();
                prev_imu_bias = imuBias::ConstantBias();

                auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6).finished());

                gtsam::PriorFactor<gtsam::Pose3> prior_factor(gtsam::Symbol('x', 0), stereo_cam.pose(), prior_noise);
                gtsam::PriorFactor<gtsam::Vector3> v_prior_factor(gtsam::Symbol('v', 0), prev_velocity, velocity_noise);
                gtsam::PriorFactor<gtsam::imuBias::ConstantBias> b_prior_factor (gtsam::Symbol('b', 0), prev_imu_bias, bias_noise);

                session_factors_[std::make_pair(sym.key(), sym.key())] = std::make_pair(gtsam::serialize(prior_factor), FactorType::PRIOR);
                session_factors_[std::make_pair(v_sym.key(), v_sym.key())] = std::make_pair(gtsam::serialize(v_prior_factor), FactorType::PRIOR);
                session_factors_[std::make_pair(b_sym.key(), b_sym.key())] = std::make_pair(gtsam::serialize(b_prior_factor), FactorType::PRIOR);

                newNodes.insert(Symbol('x', 0), stereo_cam.pose());
                newNodes.insert(Symbol('v', 0), prev_velocity);
                newNodes.insert(Symbol('b', 0), prev_imu_bias);

                graph.emplace_shared< PriorFactor<Pose3> >(Symbol('x', 0), stereo_cam.pose(), pose_noise);
                graph.emplace_shared< PriorFactor<Vector3> >(Symbol('v', 0), prev_velocity, velocity_noise);
                graph.emplace_shared< PriorFactor<imuBias::ConstantBias> >(Symbol('b', 0), prev_imu_bias, bias_noise);

                // Indicate that all node values seen in pose 0 have been seen for next iteration (landmarks)
                optimizedNodes = newNodes;
            }

            // Adding between factor
            if (add_between_factor) {
                if (pKF->mnId != 0) {
                    gtsam::Symbol sym_before('x', pKF->mnId - 1);
                    gtsam::Symbol v_sym_before('v', pKF->mnId - 1);
                    gtsam::Symbol b_sym_before('b', pKF->mnId - 1);
                    if (session_values_.exists(sym_before.key())) {
                        gtsam::Pose3 relative_pose = stereo_cam.pose().between(session_values_.at<gtsam::Pose3>(sym_before.key())).between(gtsam::Pose3());
                        gtsam::BetweenFactor<gtsam::Pose3> between_factor(sym_before, sym, relative_pose, between_factors_prior_);
                        session_factors_[std::make_pair(sym_before.key(), sym.key())] = std::make_pair(gtsam::serialize(between_factor), FactorType::BETWEEN);

                        gtsam::ImuFactor imu_factor(sym_before, v_sym_before,
                                                    sym, v_sym,
                                                    b_sym_before, pKF->mpImuPreintegrated->gtsam_imu_preintegrated);

                        gtsam::imuBias::ConstantBias zero_bias(Vector3(0, 0, 0), Vector3(0, 0, 0));

                        gtsam::BetweenFactor<gtsam::imuBias::ConstantBias> imuBiasFactor(b_sym_before,
                                                                                         b_sym_before,
                                                                                         zero_bias,
                                                                                         bias_noise);

                        graph.emplace_shared<ImuFactor>(
                                sym_before, v_sym_before,
                                sym, v_sym,
                                b_sym_before, pKF->mpImuPreintegrated->gtsam_imu_preintegrated
                        );
                        // imuBias::ConstantBias zero_bias(Vector3(0, 0, 0), Vector3(0, 0, 0));
                        graph.emplace_shared< BetweenFactor<imuBias::ConstantBias> >(
                                b_sym_before,
                                b_sym_before,
                                zero_bias,
                                bias_noise
                        );

                        // Predict initial estimates for current state
                        NavState prev_optimized_state = NavState(stereo_cam.pose(), prev_velocity);
                        NavState propagated_state = pKF->mpImuPreintegrated->gtsam_imu_preintegrated.predict(prev_optimized_state, prev_imu_bias);
                        newNodes.insert(Symbol('x', pKF->mnId), propagated_state.pose());
                        newNodes.insert(Symbol('v', pKF->mnId), propagated_state.v());
                        newNodes.insert(Symbol('b', pKF->mnId), prev_imu_bias);

                    }
                }
            }

            // Update most recent keyframe
            if ((pKF->mTimeStamp > std::get<1>(recent_kf_)) || (pKF->mnId == 0)) {
                recent_kf_ = std::make_tuple(gtsam::serialize(sym), pKF->mTimeStamp, gtsam::serialize(stereo_cam.pose()));
            }
        }

        void GTSAMOptimizer::updateLandmark(vi_slam::datastructures::MapPoint *pMP) {
            // Create landmark symbol
            gtsam::Symbol sym('l', pMP->id_);

            // Create landmark position
            cv::Mat p_cv = pMP->GetWorldPos();
            gtsam::Point3 p_gtsam(p_cv.at<float>(0), p_cv.at<float>(1), p_cv.at<float>(2));

            session_values_.insert(sym.key(), p_gtsam);

            newNodes.insert(sym.key(), p_gtsam);

            noiseModel::Isotropic::shared_ptr prior_landmark_noise = noiseModel::Isotropic::Sigma(3, 500); // 50m std on x,y,z
            graph.emplace_shared<PriorFactor<Point3> >(sym.key(), p_gtsam, prior_landmark_noise);

        }

        void GTSAMOptimizer::updateObservations(MapPoint *pMP, const std::map<KeyFrame*,std::tuple<int,int>> &observations) {
            for (const auto &mit: observations) {
                KeyFrame *pKFi = mit.first;
                if (pKFi->isBad())
                    continue;

                tuple<int,int> indexes = mit.second;
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[get<0>(indexes)];
                // Monocular observation
                if (pKFi->mvuRight[get<1>(indexes)] < 0) {
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    addMonoMeasurement(pKFi, pMP, obs, invSigma2);
                } else // Stereo observation
                {
                    Eigen::Matrix<double, 3, 1> obs;
                    const float kp_ur = pKFi->mvuRight[get<1>(indexes)];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    addStereoMeasurement(pKFi, pMP, obs, invSigma2);
                }
            }
        }

        void GTSAMOptimizer::calculateDiffrencesBetweenValueSets() {
            // Handle added states
            if (last_session_values_.empty()) {
                add_states_ = session_values_.keys();
            } else {
                add_states_ = getDifferenceKeyList(session_values_.keys(), last_session_values_.keys());
            }

            // Handle deleted states
            del_states_ = getDifferenceKeyList(last_session_values_.keys(), session_values_.keys());
        }

        void GTSAMOptimizer::calculateDiffrencesBetweenFactorSets() {
            // Handle added factors
            std::map<std::pair<gtsam::Key, gtsam::Key>, std::pair<std::string, FactorType>> add_factors_map;
            if (last_session_factors_.empty()) {
                add_factors_map = session_factors_;
                exportValuesFromMap(add_factors_map, add_factors_);
            } else {
                add_factors_map = getDifferenceSet(session_factors_, last_session_factors_);
                exportValuesFromMap(add_factors_map, add_factors_);
            }

            // Handle deleted factors
            std::map<std::pair<gtsam::Key, gtsam::Key>, std::pair<std::string, FactorType>>
                    del_factors_map = getDifferenceSet(last_session_factors_, session_factors_);
            exportKeysFromMap(del_factors_map, del_factors_);
        }

        gtsam::KeyVector GTSAMOptimizer::getDifferenceKeyList(const gtsam::KeyVector &vec_A, const gtsam::KeyVector &vec_B) {
            gtsam::KeyVector diff_vec;
            for (const auto &it_A: vec_A) {
                if (std::find(vec_B.begin(), vec_B.end(), it_A) == vec_B.end()) {
                    diff_vec.push_back(it_A);
                }
            }
            return diff_vec;
        }

        void GTSAMOptimizer::setUpdateType(const vi_slam::optimization::GTSAMOptimizer::UpdateType update_type) {
            update_type_ = update_type;
            if (update_type_ == BATCH) {
                logger_->info("setUpdateType - Batch");
            } else if (update_type_ == INCREMENTAL) {
                logger_->info("setUpdateType - Incremental");
            }
        }
    }
}

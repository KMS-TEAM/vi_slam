//
// Created by cit-industry on 11/07/2021.
//

#include "vi_slam/optimization/iSAM2Optimizer.h"
#include "vi_slam/basics/yaml.h"

namespace vi_slam{
    namespace optimization{

        using namespace datastructures;
        using namespace gtsam;
        using namespace std;

        iSAM2Optimizer::iSAM2Optimizer(std::string& config_path) {

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
        }

        iSAM2Optimizer::~iSAM2Optimizer(){};

        void iSAM2Optimizer::initializeIMUParameters(Tracking &tracker) {
            tracker.PreintegrateIMU();
        }

        void iSAM2Optimizer::optimizerVIO_IMU() {

        }


    }
}
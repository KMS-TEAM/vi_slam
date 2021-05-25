//
// Created by lacie on 25/05/2021.
//

#ifndef VI_SLAM_CONFIG_H
#define VI_SLAM_CONFIG_H

#include "../common_include.h"

namespace vi_slam{
    namespace basics{

        /// Interface with config file
        class Config {

        public:
            /// Set a new config file
            static void setParameterFile(const std::string &filename);

            /// Get a content by key
            template<typename T>
            static T get(const std::string &key);

            /// Get a vector of content by key
            template<typename T>
            static std::vector<T> getVector(const std::string &key);

            /** Get a content by key
             * The content is of type string ("true", "false").
             * It's then convertd to bool and returned.
             */
            static bool getBool(const std::string &key);

            ~Config();

        private:
            static std::shared_ptr<Config> config_;
            cv::FileStorage file_;

            /** @brief: Get content by key
             * If key doesn't exist, throw runtime error;
             */
            static cv::FileNode get_(const std::string &key);

            Config(){}
        };

        /// Get a content of cv::FileNode. Convert to type T
        template<typename T>
        T Config::get(const std::string &key) {
            cv::FileNode content = Config::get_(key);
            return static_cast<T>(content);
        }

        /// Get a content of cv::FileNode. Convert to type vector<T>
        template<typename T>
        std::vector<T> Config::getVector(const std::string &key) {
            cv::FileNode content = Config::get_(key);
            std::vector<T> res;
            content >> res;
            return res;
        }
    }
}



#endif //VI_SLAM_CONFIG_H

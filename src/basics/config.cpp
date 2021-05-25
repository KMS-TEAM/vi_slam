//
// Created by lacie on 25/05/2021.
//

#include "vi-slam/config.h"

namespace mono_vo{
    namespace basics{
        std::shared_ptr<Config> Config::config_ = nullptr;

        void Config::setParameterFile(const std::string &filename) {
            if(config_ == nullptr){
                config_ = std::shared_ptr<Config>(new Config);
            }

            cv::FileStorage *fs = new cv::FileStorage(filename, cv::FileStorage::READ);
            if(fs->isOpened() == false){
                std::cerr << "Parameter file " << filename << " does not exist." << std::endl;
                return;
            }

            config_->file_ = *fs;
        }

        Config::~Config() {
            if(file_.isOpened()){
                file_.release();
            }
        }

        cv::FileNode Config::get_(const std::string &key) {
            cv::FileNode content = Config::config_->file_[key];
            if(content.empty()){
                throw std::runtime_error("Key " + key + " doesn't eixt");
            }
            return content;
        }

        bool Config::getBool(const std::string &key) {
            std::string val = static_cast<std::string>(Config::get_(key));
            if(val == "true" || val == "True"){
                return true;
            }
            else return false;
        }
    }
}

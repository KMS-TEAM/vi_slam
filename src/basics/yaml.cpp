//
// Created by lacie on 25/05/2021.
//

#include "vi_slam/basics/yaml.h"

namespace vi_slam{
    namespace basics{
        Yaml::Yaml(const std::string &filename) : file_storage_(filename, cv::FileStorage::READ),
                                                  is_file_node_(false){
            if (!file_storage_.isOpened()){
                std::cout << "Error reading config file: " << filename << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        Yaml::Yaml(const cv::FileNode &fn) : file_node_(fn),
                                             is_file_node_(true){
        }

        Yaml::Yaml(const cv::FileStorage &fs) : file_storage_(fs),
                                                is_file_node_(false){
        }

        void Yaml::Release(){
            if (!is_file_node_ && file_storage_.isOpened()) {
                file_storage_.release();
            }
        }

        Yaml::~Yaml(){
            this->Release();
        }

        Yaml Yaml::get(const std::string &key) const{
            cv::FileNode content = get_(key);
            return Yaml(content);
        }

        cv::FileNode Yaml::get_(const std::string &key) const{
            cv::FileNode content;
            if (is_file_node_) {
                content = file_node_[key];
            }
            else
                content = file_storage_[key];
            if (content.empty()){
                std::cout << "Error: key '" << key << "' doesn't exist" << std::endl;
                exit(EXIT_FAILURE);
            }
            return content;
        }
    }
}

//
// Created by cit-industry on 10/07/2021.
//

#ifndef VI_SLAM_LOGGER_H
#define VI_SLAM_LOGGER_H

#include <iostream>
#include <fstream>

#define LOG_FILE "log.txt"
#define DEBUG(str) Logger::instance()->writeLog(str)
#define LOG Logger::instance()->output() << "[" << __FUNCTION__ << "][" << __LINE__ << "] "

namespace vi_slam{
    namespace basics{
        class Logger {
            static Logger* m_instance;
            static std::fstream outFile;
        public:
            static Logger* instance();
            ~Logger();
            static std::fstream& output();
            static void writeLog(const std::string& msg);
        private:
            Logger();
            Logger(const Logger& _other) = delete;
            void operator= (const Logger& _other) = delete;
        };
    }
}

#endif //VI_SLAM_LOGGER_H

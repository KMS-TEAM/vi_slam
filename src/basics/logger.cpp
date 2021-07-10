//
// Created by cit-industry on 10/07/2021.
//

#include "vi_slam/basics/logger.h"

namespace vi_slam{
    namespace basics{
        using namespace std;

        Logger* Logger::m_instance = nullptr;
        std::fstream Logger::outFile = fstream();

        Logger::Logger()
        {
            outFile.open(LOG_FILE, ios::out | ios::app);
        }

        Logger::~Logger()
        {
            if (nullptr != m_instance)
            {
                delete m_instance;
            }
            m_instance = nullptr;
            outFile.close();
        }

        Logger* Logger::instance()
        {
            if (nullptr == m_instance)
            {
                m_instance = new Logger();
            }
            return m_instance;
        }

        std::fstream& Logger::output()
        {
            outFile.flush();
            return outFile;
        }

        void Logger::writeLog(const std::string& msg)
        {
            outFile << "[Msg]" << msg;
            outFile.flush();
        }

    }
}



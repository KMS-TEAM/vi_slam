file(GLOB_RECURSE SOURCES
        basics/*.cpp
        datastructures/*.cpp
        display/*.cpp
        geometry/*.cpp
        optimization/*.cpp
        core/*cpp
        )

add_library(vi_slam SHARED ${SOURCES})

target_link_libraries( vi_slam
        #vilib::vilib
        ${THIRD_PARTY_LIBS}
        )


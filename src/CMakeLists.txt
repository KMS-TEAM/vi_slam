file(GLOB_RECURSE SOURCES

        datastructures/*.cpp
        display/*.cpp
        geometry/*.cpp
        optimization/*.cpp
        core/*cpp
        )

add_library(basics SHARED
        basics/basics.cpp
        basics/config.cpp
        basics/converter.cpp
        basics/eigen_funcs.cpp
        basics/opencv_funcs.cpp
        basics/yaml.cpp)

target_link_libraries(basics ${THIRD_PARTY_LIBS})

add_library(vi_slam SHARED ${SOURCES})

target_link_libraries( vi_slam
        vilib::vilib
        basics
        ${THIRD_PARTY_LIBS}
        )


#openCV4
find_package( OpenCV 4.2 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(convertVoc convertVoc.cpp )
target_link_libraries( convertVoc ${THIRD_PARTY_LIBS})
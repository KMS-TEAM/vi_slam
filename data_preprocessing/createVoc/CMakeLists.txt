find_package( OpenCV 4.2 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( createVoc createVoc.cpp )
target_link_libraries( createVoc ${THIRD_PARTY_LIBS})
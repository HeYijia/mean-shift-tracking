cmake_minimum_required(VERSION 2.8)
project(meanshift)

#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

SET(CMAKE_BUILD_TYPE Debug)

SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

find_package(OpenCV 2.4 REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})

add_library(meanshiftlib meanshift.cpp meanshift.h)
target_link_libraries(meanshiftlib ${OpenCV_LIBS})
add_executable(meanshift main.cpp)
target_link_libraries(meanshift  meanshiftlib )

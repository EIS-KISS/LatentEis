cmake_minimum_required(VERSION 3.10)

project(latenteis C CXX)


set(COMMON_COMPILE_OPTIONS
	"-Wall"
	"-Wno-reorder"
	"-Wfatal-errors"
	"-ffast-math"
	"-ftree-vectorize"
	"-g"
	"-fno-strict-aliasing")

set(SRC
	src/tokenize.cpp
	src/tokenize.h
	src/nanoflann.h
	src/main.cpp)
message(${SRC})

add_executable(${PROJECT_NAME} ${SRC})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
install(TARGETS ${PROJECT_NAME} DESTINATION bin)

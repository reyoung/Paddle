cmake_minimum_required(VERSION 3.11)
project(dlmopen)

set(CMAKE_CXX_STANDARD 11)

add_executable(dlmopen main.cpp)
target_link_libraries(dlmopen -ldl)

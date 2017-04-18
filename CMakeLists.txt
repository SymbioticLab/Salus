cmake_minimum_required(VERSION 2.6)
project(executor)

add_executable(executor main.cpp)

install(TARGETS executor RUNTIME DESTINATION bin)

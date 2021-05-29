cmake_minimum_required (VERSION 3.19)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project ("ai" CXX)

add_subdirectory(src)
add_subdirectory(test)

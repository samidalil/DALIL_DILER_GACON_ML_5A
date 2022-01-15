cmake_minimum_required(VERSION 3.20)
project(Atlus)

set(CMAKE_CXX_STANDARD 20)

#add_library(Atlus SHARED "headers/MLPData.h" "headers/core.h" "src/core.cpp" "headers/utils.h" "src/utils.cpp")
add_executable(Atlus "src/tests.cpp"  "headers/MLPData.h" "headers/core.h" "src/core.cpp" "headers/utils.h" "src/utils.cpp")

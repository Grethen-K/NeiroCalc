cmake_minimum_required(VERSION 3.20)
project(neiro CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Оптимизации + предупреждения (MSVC и GCC/Clang)
if(MSVC)
    add_compile_options(/O2 /W4)
else()
    add_compile_options(-O3 -Wall -Wextra)
endif()

# OpenMP (для будущих фаз ускорения)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable(neiro
    src/main.cpp
    src/layer.cpp
    src/network.cpp
    src/dataset.cpp
)

find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(neiro PRIVATE Eigen3::Eigen)
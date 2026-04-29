cmake_minimum_required(VERSION 3.20)
project(osmiArrow)

add_compile_options(-O3)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED ENV{MKLROOT})
    set(ENV{MKLROOT} "$ENV{HOME}/intel/oneapi/mkl/2026.0")
endif()
set(MKLROOT $ENV{MKLROOT})
message(STATUS "MKLROOT: ${MKLROOT}")

find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)

add_library(osmi_arrow INTERFACE)
target_include_directories(osmi_arrow INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${MKLROOT}/include
)
target_link_libraries(osmi_arrow INTERFACE
        Arrow::arrow_shared
        Parquet::parquet_shared
)

add_executable(osmiArrow main.cpp)

set(IOMP5_LIB "$ENV{HOME}/intel/oneapi/compiler/latest/lib/libiomp5.so")

target_link_libraries(osmiArrow PRIVATE
        osmi_arrow
        mkl_rt
        ${IOMP5_LIB}
        pthread
        m
        dl
)

target_link_directories(osmiArrow PRIVATE
        ${MKLROOT}/lib
)

target_link_options(osmiArrow PRIVATE
        -Wl,--no-as-needed
        -Wl,-rpath,${MKLROOT}/lib
        -Wl,-rpath,$ENV{HOME}/intel/oneapi/compiler/latest/lib
)
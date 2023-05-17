project(mesh_tools)
cmake_minimum_required(VERSION 3.16)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# alembic
include(ExternAlembic)

# openexr
find_package(ECM REQUIRED NO_MODULE) # ships `FindOpenEXR.cmake` (requires `extra-cmake-modules` package)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${ECM_FIND_MODULE_DIR})
find_package(OpenEXR REQUIRED)

# CLI11
if(NOT DEFINED CLI11_DIR)
  set (CLI11_DIR "${CMAKE_SOURCE_DIR}/third_party/CLI11" CACHE STRING "CLI11 git repository")
endif()
include_directories(${CLI11_DIR}/include)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

OPTION(BUILD_RENDERER "Build Alembic renderer" ON)
if(BUILD_RENDERER)
  add_subdirectory(renderer)
endif()

OPTION(BUILD_EXTRACTOR "Build Alembic extractor" ON)
if(BUILD_EXTRACTOR)
  add_subdirectory(alembic_extractor)
endif()

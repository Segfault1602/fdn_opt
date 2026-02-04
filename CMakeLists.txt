cmake_minimum_required(VERSION 4.1.1)

project(
  FdnOpt
  VERSION 0.0.1
  DESCRIPTION "FDN optimization library")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(FdnOpt_ENABLE_HARDENING "Enable hardening" ON)
option(FdnOpt_USE_SANITIZER "Use AddressSanitizer" OFF)
option(FDN_OPT_BUILD_APPS "Build applications" ${PROJECT_IS_TOP_LEVEL})

include(cmake/project-warnings.cmake)
include(cmake/project-options.cmake)

include(fetch_deps.cmake)

if(PROJECT_IS_TOP_LEVEL)
  set(CMAKE_COMPILE_WARNING_AS_ERROR ON)
endif()

add_subdirectory(src)

if(FDN_OPT_BUILD_APPS)
  message(STATUS "Building FdnOpt applications")
  add_subdirectory(app)
endif()

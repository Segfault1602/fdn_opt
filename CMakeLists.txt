cmake_minimum_required(VERSION 4.1.1)

project(
  PROJECT_NAME
  VERSION 0.0.1
  DESCRIPTION "PROJECT_DESCRIPTION")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(myproject_ENABLE_HARDENING "Enable hardening" ON)
option(myproject_USE_SANITIZER "Use AddressSanitizer" ON)

include(cmake/project-warnings.cmake)
include(cmake/project-options.cmake)

if(PROJECT_IS_TOP_LEVEL)
  set(CMAKE_COMPILE_WARNING_AS_ERROR ON)
endif()

add_library(myproject_warnings INTERFACE)
add_library(myproject::myproject_warnings ALIAS myproject_warnings)
target_compile_options(myproject_warnings INTERFACE ${MYPROJECT_WARNINGS_CXX})

add_subdirectory(src)
add_subdirectory(app)

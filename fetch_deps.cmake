# Dependency that should already be installed on the system:
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(OpenMP)

include(FetchContent)

# Below is the list of third-party dependencies fetched via CPM.cmake

FetchContent_Declare(
  CPM
  GIT_REPOSITORY https://github.com/cpm-cmake/CPM.cmake
  GIT_TAG v0.42.1)
FetchContent_MakeAvailable(CPM)
include(${cpm_SOURCE_DIR}/cmake/CPM.cmake)

cpmaddpackage("gh:odygrd/quill@11.0.2")

cpmaddpackage(
  NAME
  armadillo
  GIT_TAG
  15.2.2
  GIT_REPOSITORY
  https://gitlab.com/conradsnicta/armadillo-code.git
  DOWNLOAD_ONLY
  TRUE)

if(armadillo_ADDED)
  set(ARMADILLO_INCLUDE_DIR
      ${armadillo_SOURCE_DIR}/include
      CACHE PATH "Armadillo include directory")
  add_library(armadillo INTERFACE)
  target_include_directories(armadillo INTERFACE ${armadillo_SOURCE_DIR}/include)
  target_link_libraries(armadillo INTERFACE BLAS::BLAS LAPACK::LAPACK)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(armadillo INTERFACE OpenMP::OpenMP_CXX)
    target_compile_definitions(armadillo INTERFACE ARMA_USE_OPENMP)
  endif()
else()
  message(FATAL_ERROR "Armadillo package not added correctly")
endif()
add_library(Armadillo::Armadillo ALIAS armadillo)

cpmaddpackage(
  NAME
  ensmallen
  GIT_REPOSITORY
  https://github.com/mlpack/ensmallen.git
  GIT_TAG
  master
  DOWNLOAD_ONLY
  TRUE)

if(ensmallen_ADDED)
  add_library(ensmallen INTERFACE IMPORTED)
  target_include_directories(ensmallen INTERFACE ${ensmallen_SOURCE_DIR}/include)
  set(ENSMALLEN_INCLUDE_DIR
      ${ensmallen_SOURCE_DIR}/include
      CACHE PATH "Ensmallen include directory")
else()
  message(FATAL_ERROR "Ensmallen package not added correctly")
endif()

cpmaddpackage(
  URI
  "gh:Segfault1602/audio_utils#main"
  OPTIONS
  "AUDIO_UTILS_USE_RTAUDIO OFF"
  "AUDIO_UTILS_ENABLE_HARDENING ON"
  "AUDIO_UTILS_USE_SANITIZER OFF")

cpmaddpackage(
  NAME
  sfFDN
  GIT_REPOSITORY
  https://github.com/Segfault1602/sfFDN.git
  GIT_TAG
  main)

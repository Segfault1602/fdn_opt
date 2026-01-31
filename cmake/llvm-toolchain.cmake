include_guard(GLOBAL)

if(APPLE)
  set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm@20/bin/clang++")
  set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm@20/bin/clang")
else()
  set(CMAKE_CXX_COMPILER "clang++")
  set(CMAKE_C_COMPILER "clang")
endif()

# For openmp support on Windows
if(WIN32)
  set(CMAKE_SYSTEM_LIBRARY_PATH "$ENV{ProgramFiles}/LLVM/lib")
endif()

include($ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)

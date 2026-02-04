if(MSVC)
  set(FdnOpt_WARNINGS_CXX /W3 /permissive-)
elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
  set(FdnOpt_WARNINGS_CXX
      -Wall
      -Wextra
      -Wpedantic
      -Wno-sign-compare
      -Wno-language-extension-token)

endif()

add_library(FdnOpt_warnings INTERFACE)
add_library(FdnOpt::FdnOpt_warnings ALIAS FdnOpt_warnings)
target_compile_options(FdnOpt_warnings INTERFACE ${FdnOpt_WARNINGS_CXX})

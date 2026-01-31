add_library(myproject.foo)
add_library(myproject::foo ALIAS myproject.foo)

target_sources(myproject.foo PRIVATE foo.cpp)
target_sources(
  myproject.foo
  PUBLIC FILE_SET
         HEADERS
         BASE_DIRS
         ${CMAKE_CURRENT_SOURCE_DIR}/../include
         FILES
         ${CMAKE_CURRENT_SOURCE_DIR}/../include/myproject/foo.h)

target_link_libraries(myproject.foo PRIVATE myproject::myproject_options myproject::myproject_warnings)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE myproject::myproject_options myproject::myproject_warnings myproject.foo)

cpmaddpackage("gh:CLIUtils/CLI11@2.6.1")

add_library(fdn_opt_cli STATIC cli_options.cpp cli_options.h optimizer_cli.cpp optimizer_cli.h)
target_include_directories(fdn_opt_cli PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
  fdn_opt_cli
  PUBLIC fdn_optimization
         sfFDN.sfFDN
         quill::quill
         audio_utils::audio_utils
         CLI11::CLI11
  PRIVATE FdnOpt::FdnOpt_options
          FdnOpt::FdnOpt_warnings)

add_library(
  fdn_opt_application STATIC
  application.cpp
  application.h
  optimization_workflows.cpp
  optimization_workflows.h
  utils.h)
target_include_directories(fdn_opt_application PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
  fdn_opt_application
  PUBLIC fdn_opt_cli
  PRIVATE fdn_optimization
          sfFDN.sfFDN
          quill::quill
          audio_utils::audio_utils
          Armadillo::Armadillo
          nlohmann_json::nlohmann_json
          FdnOpt::FdnOpt_options
          FdnOpt::FdnOpt_warnings)

add_executable(optim_tool main.cpp utils.h)
target_link_libraries(
  optim_tool
  PRIVATE fdn_opt_application
          fdn_opt_cli
          fdn_optimization
          sfFDN.sfFDN
          quill::quill
          audio_utils::audio_utils
          Armadillo::Armadillo
          nlohmann_json::nlohmann_json
          FdnOpt::FdnOpt_options
          FdnOpt::FdnOpt_warnings)

if(OpenMP_CXX_FOUND)
  target_link_libraries(optim_tool PRIVATE OpenMP::OpenMP_CXX)
endif()

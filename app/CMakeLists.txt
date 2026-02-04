cpmaddpackage("gh:CLIUtils/CLI11@2.6.1")

add_executable(optim_tool main.cpp utils.h)
target_link_libraries(
  optim_tool
  PRIVATE fdn_optimization
          sfFDN.sfFDN
          quill::quill
          audio_utils::audio_utils
          Armadillo::Armadillo
          CLI11::CLI11
          FdnOpt::FdnOpt_options
          FdnOpt::FdnOpt_warnings)

if(OpenMP_CXX_FOUND)
  target_link_libraries(optim_tool PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(tuning tuning.cpp)
target_link_libraries(
  tuning
  PRIVATE fdn_optimization
          sfFDN.sfFDN
          quill::quill
          audio_utils::audio_utils
          Armadillo::Armadillo
          FdnOpt::FdnOpt_options
          FdnOpt::FdnOpt_warnings)

if(OpenMP_CXX_FOUND)
  target_link_libraries(tuning PRIVATE OpenMP::OpenMP_CXX)
endif()

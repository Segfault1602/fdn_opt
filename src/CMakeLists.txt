add_library(fdn_optimization STATIC)
target_sources(
  fdn_optimization
  PRIVATE optimizer.cpp
          model.cpp
          loss.cpp
          random_searcher.cpp
  PUBLIC FILE_SET
         "fdn_optimization_headers"
         TYPE
         HEADERS
         BASE_DIRS
         ./../include/fdn_opt
         FILES
         ./../include/fdn_opt/optimizer.h
         ./../include/fdn_opt/model.h
         ./../include/fdn_opt/loss.h
         ./../include/fdn_opt/optim_types.h)

target_link_libraries(
  fdn_optimization
  PRIVATE audio_utils::audio_utils
          sfFDN.sfFDN
          quill::quill
          Armadillo::Armadillo
          ensmallen
          FdnOpt::FdnOpt_options
          FdnOpt::FdnOpt_warnings)

if(OpenMP_CXX_FOUND)
  target_link_libraries(fdn_optimization PRIVATE OpenMP::OpenMP_CXX)
endif()

target_include_directories(fdn_optimization SYSTEM PRIVATE ${ensmallen_SOURCE_DIR}/include)

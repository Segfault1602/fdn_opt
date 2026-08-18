#include "application.h"
#include "cli_options.h"

#include <CLI/CLI.hpp>
#include <quill/Backend.h>
#include <quill/Frontend.h>
#include <quill/Logger.h>
#include <quill/sinks/ConsoleSink.h>

#include <iostream>
#include <utility>

int main(int argc, char** argv)
{
    quill::Backend::start();
    quill::Logger* logger = quill::Frontend::create_or_get_logger(
        "root", quill::Frontend::create_or_get_sink<quill::ConsoleSink>("sink_id_1"));

    CLI::App app{"FDN Optimization Tool"};
    fdn_opt_app::RawCliOptions raw_options;
    fdn_opt_app::ConfigureCliApp(app, raw_options);
    CLI11_PARSE(app, argc, argv);
    if (!raw_options.parsed)
    {
        std::cerr << "CLI options were not finalized.\n";
        return -1;
    }

    fdn_opt_app::OptimizationApplication application(logger, std::move(*raw_options.parsed));
    return application.Run();
}

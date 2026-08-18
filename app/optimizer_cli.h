#pragma once

#include "cli_options.h"

namespace fdn_opt_app
{

void RegisterOptimizerSubcommands(CLI::App& app, OptimizerCliOptions& options);

} // namespace fdn_opt_app

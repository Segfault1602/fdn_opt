clearvars;
addpath(genpath("../../FDNToolbox"));

show_previous_run = true;

OUT_DIR = "../optim_output";

RESULTS = dir(OUT_DIR);

% Ignore "." and ".."  
RESULTS = RESULTS(3:end);

RESULTS = RESULTS([RESULTS.isdir]);
[~,idx] = sort([RESULTS.datenum]);
RESULTS = RESULTS(idx);


LATEST_DIR = RESULTS(end);
PREVIOUS_DIR = RESULTS(end-1);

INIT_CONFIG = fullfile(LATEST_DIR.folder, LATEST_DIR.name, "initial_fdn_config.txt");
COLORLESS_CONFIG = fullfile(LATEST_DIR.folder, LATEST_DIR.name, "colorless_fdn_config.txt");
PREVIOUS_CONFIG = fullfile(PREVIOUS_DIR.folder, PREVIOUS_DIR.name, "colorless_fdn_config.txt");

% compute

res_out = {};
pol_out = {};

configs = [INIT_CONFIG COLORLESS_CONFIG];
if show_previous_run
    configs(3) = PREVIOUS_CONFIG;
end

parfor n = 1:length(configs)
   [res, pol] = GetPolesAndResidueFromConfig(configs(n));
   % res = res / max(abs(res));
   res = mag2db(abs(res));
  
   % Remove data where res < 150dB
   pol(res < -150) = [];  % Corresponding poles are also removed
   res(res < -150) = [];  % Remove data where res < 150dB

   res_out{n} =  res;
   pol_out{n} = pol;
end

% [res, pol] = GetPolesAndResidueFromConfig(INIT_CONFIG);
% [res_opt, pol_opt] = GetPolesAndResidueFromConfig(COLORLESS_CONFIG);
% [res_prev, pol_prev] = GetPolesAndResidueFromConfig(PREVIOUS_CONFIG);
res = res_out{1};
pol = pol_out{1};
res_opt = res_out{2};
pol_opt = pol_out{2};

if show_previous_run
    res_prev = res_out{3};
    pol_prev = pol_out{3};
end

% plot
figure(1); clf;grid on;

plot(angle(pol), mag2db(abs(pol)),'.', DisplayName="Init");

hold on;
plot(angle(pol_opt), mag2db(abs(pol_opt)),'.', DisplayName="Optim");
hold off;
legend();
% legend({'Poles','Minimum Boundary', 'Maximum Boundary'});
xlabel('Pole Angle [rad]')
ylabel('Pole mag [db]')

figure(2); grid on;

plot(angle(pol),res,'.', DisplayName="Init");


hold on;
plot(angle(pol_opt), res_opt,'.', DisplayName="Optim");

hold off;
legend();
% legend({'Poles','Minimum Boundary', 'Maximum Boundary'});
xlabel('Res Angle [rad]')
ylabel('Res mag [db]')

figure(3);

num_row = 2;
if show_previous_run
    num_row = 3;
end

normalization = "pdf";

tiledlayout(num_row,1);
ax1 = nexttile;
h1 = histogram(ax1, res, DisplayName="Initial FDN", Normalization=normalization);
title("Init");

ax2 = nexttile;
h2 = histogram(ax2, res_opt, DisplayName="Optimized FDN", Normalization=normalization);
title(LATEST_DIR.name, Interpreter='none')

if show_previous_run
    ax3 = nexttile;
    h3 = histogram(ax3, res_prev, DisplayName="Previous Optimization", Normalization=normalization);
    title(PREVIOUS_DIR.name + " (Previous)", Interpreter='none');
    h3.BinWidth = 1;
end

% h1.Normalization = 'probability';
h1.BinWidth = 2;
% h2.Normalization = 'probability';
h2.BinWidth = 2;


if show_previous_run
    linkaxes([ax1 ax2 ax3], "xy");
else
    linkaxes([ax1 ax2], "xy");
end

figure(4);
tiledlayout(2,1);

ax1 = nexttile;
histogram(res, Normalization="pdf", BinWidth=2);

offset = abs(max(res)) - 1;
bhat = raylfit(-res - offset);
hold on;
x = 0:0.1:max(-res);
y = raylpdf(x, bhat);
plot(-x - offset,y,'r');
hold off;
title_string = sprintf("Initial, sigma=%f", bhat);
title(title_string);

ax2 = nexttile;
histogram(res_opt, Normalization="pdf", BinWidth=2);

offset = abs(max(res_opt)) - 1;
bhat = raylfit(-res_opt - offset);
hold on;
x = 0:0.1:max(-res_opt);
y = raylpdf(x, bhat);
plot(-x - offset,y,'r');
hold off;
title_string = sprintf("Optimized, sigma=%f", bhat);
title(title_string);

 linkaxes([ax1 ax2], "xy");
 xlim([-150 -50]);

function [res, pol] = GetPolesAndResidueFromConfig(fdn_config)
    direct = zeros(1,1);
    [A_init, B_init, C_init, m_init] = ReadFDNConfig(fdn_config);
      
    N = length(B_init);
    
    gainPerSample = db2mag(RT602slope(3, 48000));
    gainPerSamples = (gainPerSample.^m_init).';
    A_init = A_init * diag(gainPerSamples);    

    [res, pol, ~, ~] = dss2pr(m_init, A_init, B_init, C_init, direct);
end
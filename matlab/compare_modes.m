clearvars;
close all;
addpath(genpath("../../FDNToolbox"));

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

configs = [INIT_CONFIG COLORLESS_CONFIG PREVIOUS_CONFIG];

parfor n = 1:length(configs)
   [res, pol] = GetPolesAndResidueFromConfig(configs(n));
   res_out{n} = res;
   pol_out{n} = pol;
end

% [res, pol] = GetPolesAndResidueFromConfig(INIT_CONFIG);
% [res_opt, pol_opt] = GetPolesAndResidueFromConfig(COLORLESS_CONFIG);
% [res_prev, pol_prev] = GetPolesAndResidueFromConfig(PREVIOUS_CONFIG);
res = res_out{1};
pol = pol_out{1};
res_opt = res_out{2};
pol_opt = pol_out{2};
res_prev = res_out{3};
pol_prev = pol_out{3};

% plot
figure(1); grid on;

plot(angle(pol), mag2db(abs(pol)),'.', DisplayName="Init");

hold on;
plot(angle(pol_opt), mag2db(abs(pol_opt)),'.', DisplayName="Optim");
hold off;
legend();
% legend({'Poles','Minimum Boundary', 'Maximum Boundary'});
xlabel('Pole Angle [rad]')
ylabel('Pole mag [db]')

figure(2); grid on;

plot(angle(pol), mag2db(abs(res)),'.', DisplayName="Init");


hold on;
plot(angle(pol_opt), mag2db(abs(res_opt)),'.', DisplayName="Optim");

hold off;
legend();
% legend({'Poles','Minimum Boundary', 'Maximum Boundary'});
xlabel('Res Angle [rad]')
ylabel('Res mag [db]')

figure(3);


tiledlayout(3,1);
ax1 = nexttile;
h1 = histogram(ax1, mag2db(abs(res)), DisplayName="Initial FDN");
title("Init");

ax2 = nexttile;
h2 = histogram(ax2, mag2db(abs(res_opt)), DisplayName="Optimized FDN");
title(LATEST_DIR.name, Interpreter='none')

ax3 = nexttile;
h3 = histogram(ax3, mag2db(abs(res_prev)), DisplayName="Previous Optimization");
title(PREVIOUS_DIR.name + " (Previous)", Interpreter='none');

% h1.Normalization = 'probability';
h1.BinWidth = 1;
% h2.Normalization = 'probability';
h2.BinWidth = 1;
h3.BinWidth = 1;

linkaxes([ax1 ax2 ax3], "xy");

function [res, pol] = GetPolesAndResidueFromConfig(fdn_config)
    direct = zeros(1,1);
    [A_init, B_init, C_init, m_init] = ReadFDNConfig(fdn_config);
      
    N = length(B_init);
    
    gainPerSample = db2mag(RT602slope(10, 48000));
    gainPerSamples = (gainPerSample.^m_init).';
    
    absorption.a = zeros(N,1,2);
    absorption.a(:,1,1) = 1;
    absorption.b = zeros(N,1,2);
    absorption.b(:,1,1) = gainPerSamples;
    absorption.b(:,1,2) = 0;
    zAbsorption = zTF(absorption.b,absorption.a,'isDiagonal',true);
    

    [res, pol, ~, ~] = dss2pr(m_init, A_init, B_init, C_init, direct, 'absorptionFilters', zAbsorption);
end
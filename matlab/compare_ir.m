clearvars;close all;
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

INIT_IR = fullfile(LATEST_DIR.folder, LATEST_DIR.name, "colorless_initial_ir.wav");
COLORLESS_IR = fullfile(LATEST_DIR.folder, LATEST_DIR.name, "colorless_ir.wav");
PREVIOUS_IR = fullfile(PREVIOUS_DIR.folder, PREVIOUS_DIR.name, "colorless_ir.wav");

fprintf("Latest:   %s\n", LATEST_DIR.name);
fprintf("Previous: %s\n", PREVIOUS_DIR.name);

[init_ir, init_fs] = audioread(INIT_IR);
[opt_ir, opt_fs] = audioread(COLORLESS_IR);

irLen = size(init_ir,1);

noise = randn(irLen,1);
noise = noise / max(abs(noise));

figure(1);
tiledlayout(2,1);

ax1 = nexttile;
plot(ax1, init_ir);
title('Initial Impulse Response');
xlabel('Samples');
ylabel('Amplitude');

ax2 = nexttile;
plot(ax2, opt_ir);
title('Optimized Impulse Response');
xlabel('Samples');
ylabel('Amplitude');

linkaxes([ax1 ax2], "xy");


WIN_LEN = 2^13;
OVL_LEN = round(0.5*WIN_LEN);

figure(2);
spectralFlatness(init_ir, init_fs, Window=rectwin(WIN_LEN), OverlapLength=OVL_LEN, ...
              Range=[0,init_fs/2]);

init_flat = spectralFlatness(init_ir, init_fs, Window=rectwin(size(init_ir,1)), OverlapLength=OVL_LEN, ...
              Range=[0,init_fs/2]);

hold on;
spectralFlatness(opt_ir, opt_fs, Window=rectwin(WIN_LEN), OverlapLength=OVL_LEN, ...
              Range=[0,opt_fs/2]);
optim_flat = spectralFlatness(opt_ir, opt_fs, Window=rectwin(size(opt_ir,1)), OverlapLength=OVL_LEN, ...
              Range=[0,opt_fs/2]);

spectralFlatness(noise, opt_fs, Window=rectwin(WIN_LEN), OverlapLength=OVL_LEN, ...
              Range=[0,opt_fs/2]);
noise_flat = spectralFlatness(noise, opt_fs, Window=rectwin(size(noise,1)), OverlapLength=OVL_LEN, ...
              Range=[0,opt_fs/2]);

yline(init_flat, "b--", DisplayName="Initial Flatness");
yline(optim_flat, "r--", DisplayName="Optimized Flatness");
yline(noise_flat, "k--", DisplayName="Noise Flatness");

hold off;
ylim([0 1]);
legend("initial", "optimized", "Noise");
title("Spectral Flatness");

figure(3);
sparsity_win = 2^13;
sparsity_ovl = round(0.9*sparsity_win);
[sparsity_losses, init_sparsity_loss] = SparsityLoss(init_ir, sparsity_win, sparsity_ovl);
plot(sparsity_losses, DisplayName="Initial IR");
yline(init_sparsity_loss, "b--", DisplayName="Initial IR");

hold on;
[sparsity_losses_opt, total_sparsity_loss_opt] = SparsityLoss(opt_ir, sparsity_win, sparsity_ovl);
plot(sparsity_losses_opt, "r-", DisplayName="Optimized IR");
yline(total_sparsity_loss_opt, "r--", DisplayName="Optimized");

[sparsity_losses, noise_sparsity_loss] = SparsityLoss(noise, sparsity_win, sparsity_ovl);
plot(sparsity_losses, "k-", DisplayName="Noise");
yline(noise_sparsity_loss, "k--", DisplayName="Noise");

hold off;
legend();
title("Sparsity Loss");
grid on;


figure(4);
loss_history_file = fullfile(LATEST_DIR.folder, LATEST_DIR.name,"colorless_loss_history.txt");
losses = readtable(loss_history_file, "VariableNamingRule","preserve");

prev_lh_file = fullfile(PREVIOUS_DIR.folder, PREVIOUS_DIR.name,"colorless_loss_history.txt");
prev_losses = readtable(prev_lh_file, "VariableNamingRule","preserve");

grid_size = ceil(sqrt(size(losses,2)));
prev_grid_size = ceil(sqrt(size(prev_losses,2)));

grid_size = max(grid_size, prev_grid_size);


losses = losses(:, sort(losses.Properties.VariableNames));
prev_losses = prev_losses(:, sort(prev_losses.Properties.VariableNames));


tiledlayout(grid_size, grid_size);


ax1 = nexttile;
total_loss = losses{:,"Total"};
plot(ax1, total_loss, DisplayName="Latest");

hold on;
prev_total_loss = prev_losses{:, "Total"};
plot(ax1, prev_total_loss, "--", DisplayName="Previous");
hold off;
title("Total");
grid on;
legend();


losses = removevars(losses, "Total");
prev_losses = removevars(prev_losses, "Total");

for n = 1:size(losses,2)
    ax_n = nexttile;
    loss_name = losses.Properties.VariableNames{n};
    plot(ax_n, losses{:, n}, DisplayName="Latest");
    

    if (any(ismember(prev_losses.Properties.VariableNames, loss_name)))
        prev_loss = prev_losses{:, loss_name};
        hold on;
        plot(ax_n, prev_losses{:, n}, DisplayName="Previous");
        hold off;
    end

    title(loss_name)
    legend();
    grid on;
end


clearvars;close all;
addpath(genpath("../../FDNToolbox"));
addpath(genpath("../../DecayFitNet"));

OUT_DIR = "../optim_output";

RESULTS = dir(OUT_DIR);

% Ignore "." and ".."  
RESULTS = RESULTS(3:end);

RESULTS = RESULTS([RESULTS.isdir]);
[~,idx] = sort([RESULTS.datenum]);
RESULTS = RESULTS(idx);


LATEST_DIR = RESULTS(end);

INIT_IR = fullfile(LATEST_DIR.folder, LATEST_DIR.name, "initial_ir.wav");
OPTIM_IR = fullfile(LATEST_DIR.folder, LATEST_DIR.name, "optimized_ir.wav");
OPTIM_LOSS = fullfile(LATEST_DIR.folder, LATEST_DIR.name, "loss_history.txt");
OPTIM_FILTER = fullfile(LATEST_DIR.folder, LATEST_DIR.name, "optimized_filter_config.txt");
TARGET_RIR_NAME = fullfile(LATEST_DIR.folder, LATEST_DIR.name, "target_rir_name.txt");

rir_name = readlines(TARGET_RIR_NAME);
rir_name = fullfile("..", rir_name{1});

fprintf("Latest:   %s\n", LATEST_DIR.name);

[target_ir, target_fs] = audioread(rir_name);
[init_ir, init_fs] = audioread(INIT_IR);
[opt_ir, opt_fs] = audioread(OPTIM_IR);

if target_fs ~= init_fs
    [p,q] = rat(opt_fs/target_fs);
    target_ir = resample(target_ir,p,q );
    target_fs = opt_fs;
end

% opt_ir(500)= 0.5;

% target_ir = RemoveBeginningSilence(target_ir);
% opt_ir = RemoveBeginningSilence(opt_ir);

figure(1);

plot(target_ir + 1, DisplayName="Target");
hold on;
plot(opt_ir, DisplayName="Opt");
hold off;
legend();




target_edc = EDC(target_ir);
init_edc = EDC(init_ir);
opt_edc = EDC(opt_ir);

figure(2);
plot(init_edc, DisplayName="Init");
hold on;
plot(opt_edc, DisplayName="Optimized");
plot(target_edc, DisplayName="Target");
hold off;
title("Energy Decay Curve");
legend;

oct_bank = octaveFilterBank("1 octave", SampleRate=target_fs, FrequencyRange=[52, 16000]);

target_filtered = oct_bank(target_ir);
oct_bank.reset();
opt_filtered = oct_bank(opt_ir);

filter_freqs = (oct_bank.getCenterFrequencies);

% target_filtered = RemoveBeginningSilence(target_filtered);
% opt_filtered = RemoveBeginningSilence(opt_filtered);

target_edr = EDC(target_filtered);
opt_edr = EDC(opt_filtered);

% target_edr = target_edr - max(target_edr);
% opt_edr = opt_edr - max(opt_edr);

figure(3);

cmap = orderedcolors('gem12');

t = 0:size(opt_edr,1)-1;
t = t / opt_fs;
t = t';

tile = tiledlayout(3,3, TileSpacing="compact", Padding="compact");
axes = [];
for n = 1:9
    axes(n) = nexttile;
    
    if filter_freqs(n) < 1000
        freq_string = sprintf("%.0f Hz", round(filter_freqs(n),2,"significant"));
    else
        freq_string = sprintf("%.1f kHz", filter_freqs(n)/1000);
    end

    target_label = sprintf("Target - %s", freq_string);
    plot(t, target_edr(:,n), "--",  Color=cmap(n,:), DisplayName="Target RIR");
    hold on;
    plot(t, opt_edr(:,n), Color=cmap(n,:), DisplayName="Optimized FDN");
    hold off;
    legend;
    title(freq_string);
    grid on;
    ylabel([]);
    xlabel([]);
    yticks(-140:20:0);

    if n < 7
        xticklabels([]);
    end

    if any([2 3 5 6 8 9] == n)
        yticklabels([])
    end
end


xlabel(tile, "Time (s)");
ylabel(tile, "EDC (dB)");
ylim([-150 0]);
linkaxes(axes, "xy");
ylim([-140 0]);


% hold on;
% plot(target_edr(:,10),"--",  Color=cmap(10,:), DisplayName=sprintf("Target - %d", filter_freqs(10)));
% plot(opt_edr(:,10), Color=cmap(10,:), DisplayName=sprintf("Opt - %d", filter_freqs(10)))
% hold off;

legend;



figure(5);

n_fft = 4096;
hop_size = 128;
win_size = 1024;
ovl_len = win_size - hop_size;
win = hann(win_size);
n_mels = 132;

subplot(311);
% pspectrum(target_ir, target_fs, "spectrogram");
[tS, tF, tT] = melSpectrogram(target_ir, target_fs, Window=win, OverlapLength=ovl_len, FFTLength=n_fft, NumBands=n_mels);
S_db = 10 * log10(tS + eps);
surf(tT, tF, S_db, EdgeColor='none');
view([0,90]);
axis([tT(1) tT(end) tF(1) tF(end)])
title("Target");
colorbar;

subplot(312);
% pspectrum(opt_ir, opt_fs, "spectrogram");
[oS, oF, oT] = melSpectrogram(opt_ir, opt_fs, Window=win, OverlapLength=ovl_len, FFTLength=n_fft, NumBands=n_mels);
oS_db = 10 * log10(oS + eps);
surf(oT, oF, oS_db, EdgeColor='none');
view([0,90]);
axis([oT(1) oT(end) oF(1) oF(end)])
title("Optimized Impulse Response Spectrogram");
colorbar;

subplot(313);
S_err = abs(S_db - oS_db);
surf(oT, oF, S_err, EdgeColor='none');
view([0,90]);
axis([oT(1) oT(end) oF(1) oF(end)])
title("Error");
clim([0 20]);
colorbar;

figure(6);

subplot(311);

target_edr_mel = EDC(tS);
surf(tT, tF, target_edr_mel, EdgeColor='none');
view([0,90]);
axis([oT(1) oT(end) oF(1) oF(end)])
title("Target");
colorbar;


subplot(312);
opt_edr_mel = EDC(oS);
surf(oT, oF, opt_edr_mel, EdgeColor='none');
view([0,90]);
axis([oT(1) oT(end) oF(1) oF(end)])
title("Optimized");
colorbar;

subplot(313);
edr_err = abs(target_edr_mel - opt_edr_mel);
surf(oT, oF, edr_err, EdgeColor='none');
view([0,90]);
axis([oT(1) oT(end) oF(1) oF(end)])
title("Error");
colorbar;




figure(7);
losses = readtable(OPTIM_LOSS, "VariableNamingRule","preserve");
total_loss = losses{:,1};
plot(total_loss, DisplayName=losses.Properties.VariableNames{1});
hold on;
for n = 2:size(losses,2)
    plot(losses{:, n}, DisplayName=losses.Properties.VariableNames{n});
end
hold off;
grid on;
legend();

figure(8);


filter_mat = readmatrix(OPTIM_FILTER);
fdn_t60s = filter_mat(1,:);
fdn_freqs = filter_mat(2, :);
fdn_tc_gains = filter_mat(3,:);

net = DecayFitNetToolbox(1, target_fs, fdn_freqs);
[tVals_decayfitnet, aVals_decayfitnet, nVals_decayFitNet, normVals_decayFitNet] = net.estimateParameters(target_ir);
disp('==== DecayFitNet: Estimated T values (in seconds, T=0 indicates an inactive slope): ====') 
disp(tVals_decayfitnet)

[L, A, N] = decayFitNet2InitialLevel(tVals_decayfitnet, aVals_decayfitnet, nVals_decayFitNet, normVals_decayFitNet, target_fs, size(target_ir,1), 10);

[tVals_optimized, ~, ~, ~] = net.estimateParameters(opt_ir);

decayFitNet_Freqs = net.getFilterFrequencies();
semilogx(decayFitNet_Freqs, tVals_decayfitnet, DisplayName="DecayFitNet");

hold on;
plot(fdn_freqs, fdn_t60s, DisplayName="Optimized");
plot(decayFitNet_Freqs, tVals_optimized, DisplayName="Optimized - DecayFitNet");
hold off;
legend();
grid();
ylim([0 1.5*max([fdn_t60s(:); tVals_decayfitnet(:);tVals_optimized(:)])]);



function [processed_irs] = RemoveBeginningSilence(signals)
    processed_irs = zeros(size(signals));

    for n = 1:size(signals, 2)
        % Find the first non-silent sample
        max_sample = max(abs(signals(:,n)));
        direct_index = find(signals(:,n) >= max_sample*0.25, 1);
        % Remove silence from the signal
        processed_size = size(signals,1) - direct_index+1;
        processed_irs(1:processed_size,n) = signals(direct_index:end,n);
    end

end
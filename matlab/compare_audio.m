clearvars;

OUT_DIR = "../optim_output";

AUDIO_PREFIX = "drumloop";

RESULTS = dir(OUT_DIR);

% Ignore "." and ".."  
RESULTS = RESULTS(3:end);

RESULTS = RESULTS([RESULTS.isdir]);
[~,idx] = sort([RESULTS.datenum]);
RESULTS = RESULTS(idx);


LATEST_DIR = RESULTS(end);
LATEST_DIR = fullfile(LATEST_DIR.folder, LATEST_DIR.name);
TARGET_RIR_NAME = fullfile(LATEST_DIR, "target_rir_name.txt");
OPTIM_IR = fullfile(LATEST_DIR, "spectrum_optimized_ir.wav");

[x1, x1_fs] = audioread("sentences_05_loud.wav");

rir_name = readlines(TARGET_RIR_NAME);
rir_name = fullfile("..", rir_name{1});
[target_ir, target_fs] = audioread(rir_name);

[target_filepath, target_name, ~] = fileparts(rir_name);

target_name = split(target_name, "_");
target_name = target_name(2) + "_" + target_name(3);

[optim_ir, ~] = audioread(OPTIM_IR);
[target_early, ~] = audioread(fullfile(target_filepath, "early_reflections_filter", "early_"+target_name+".wav"));
[dry_audio, ~] = audioread(fullfile("..", "audio", AUDIO_PREFIX+".wav"));
[target_drum, target_fs] = audioread(fullfile(target_filepath, "audio_out", AUDIO_PREFIX+"_"+target_name+".wav"));
[fdn_drum, fs] = audioread(fullfile(LATEST_DIR, AUDIO_PREFIX+"_wet.wav"));

[p, q] = rat(fs / x1_fs);
x1 = resample(x1, p, q);

x1_target = conv(x1, target_ir);
x1_fdn = conv(x1, optim_ir);

x1_target = x1_target(1:3*fs);
x1_fdn = x1_fdn(1:3*fs);

figure(1);clf;
tiledlayout(2,1);
nexttile;
pspectrum(x1_target, fs, "spectrogram");
nexttile;
pspectrum(x1_fdn, fs, "spectrogram");

playblocking(audioplayer(x1_target, fs));
playblocking(audioplayer(x1_fdn, fs));

return;

dry_audio = dry_audio(1:2*fs);
target_drum = target_drum(1:2*target_fs);
fdn_drum = fdn_drum(1:2*fs);



dry_audio_early = conv(dry_audio, target_early, "full");

optim_convolved = conv(optim_ir, dry_audio_early, "full");
optim_convolved = optim_convolved(1:2*fs);
dry_audio_early = dry_audio_early(1:2*fs);

fdn_drum = 0.70*fdn_drum + 0*dry_audio_early;

% target_rms = rms(target_drum);
% fdn_rms = rms(fdn_drum);
% 
% target_drum = target_drum * (fdn_rms / target_rms);



figure(1);
tile = tiledlayout(3,1);
nexttile;
plot(dry_audio);
title("Dry audio");
ylim([-1 1]);

nexttile;
plot(target_drum);
title("Target")
ylim([-1 1]);

nexttile;
plot(fdn_drum);
title("FDN");
ylim([-1 1]);


target_drum = target_drum / max(abs(target_drum));
target_drum = target_drum * 0.5;

fdn_drum = fdn_drum / max(abs(fdn_drum));
fdn_drum = fdn_drum * 0.5;

disp("Playing target audio...");
playblocking(audioplayer(target_drum, target_fs));

disp("Playing fdn audio...");
playblocking(audioplayer(fdn_drum, fs));

% disp("Playing convolved audio");
% playblocking(audioplayer(optim_convolved, fs));
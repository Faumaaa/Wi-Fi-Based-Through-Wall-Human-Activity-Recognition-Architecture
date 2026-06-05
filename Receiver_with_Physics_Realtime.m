
% 20 model features + 10 physics features = 30 total
clear; close all; clc;

%% ================= HARDWARE SETTINGS =================
fc         = 2.4e9;
fs         = 5e6;
BW         = 56e6;
chirp_time = 1e-3;
rx_gain    = 30;
decimation = 10;
serial_rx  = '31CF3B4';

%% ================= RANGE SETTINGS =================
c             = 3e8;
max_range     = 1.0;
MIN_R  = 0.40;
samples_per_chirp = round(chirp_time * fs);
sweep_slope   = BW / chirp_time;
max_range_bin = round((2 * max_range * BW) / (c * chirp_time));

%% ================= FILE SETTINGS =================
CSV_FILE    = 'D:\FYP\realtime_features.csv';
FLAG_FILE   = 'D:\FYP\data_ready.flag';
WINDOW_SIZE = 50;
WRITE_EVERY = 10;

%% ================= INITIALIZE USRP =================
rx = comm.SDRuReceiver( ...
    'Platform',         'B210', ...
    'SerialNum',        serial_rx, ...
    'CenterFrequency',  fc, ...
    'Gain',             rx_gain, ...
    'DecimationFactor', decimation, ...
    'SamplesPerFrame',  samples_per_chirp, ...
    'OutputDataType',   'double');

tmp      = rx();
actual_n = numel(tmp);
nfft     = 2^nextpow2(actual_n);
max_range_bin = min(max_range_bin, nfft);

%% ================= REFERENCE CHIRP =================
t_chirp   = (0:actual_n-1) / fs;
f_inst    = -BW/2 + sweep_slope * t_chirp;
phase_ref = 2*pi*cumsum(f_inst)/fs;
ref_chirp = exp(1j * phase_ref(:));
ref_chirp = ref_chirp / max(abs(ref_chirp));

%% ================= CSV HEADER (30 features) =================
col_names = ['mean_magnitude,std_magnitude,max_magnitude,mean_phase,'     ...
             'std_phase,energy,range_peak,range_mean,range_std,'           ...
             'doppler_mean,doppler_std,peak_power,median_magnitude,'       ...
             'percentile_25,percentile_75,signal_entropy,zero_crossings,'  ...
             'rms_value,peak_to_avg_ratio,kurtosis_value,'                 ...
             'doppler_centroid,doppler_bandwidth,doppler_symmetry,'        ...
             'num_doppler_peaks,range_entropy,trunk_limb_ratio,'           ...
             'cadence_period,cadence_strength,inst_freq_std,snr_db'];
header = [col_names, '\n'];

fid = fopen(CSV_FILE,'w'); fprintf(fid, header); fclose(fid);

%% ================= BUFFER =================
N_FEATURES  = 30;
buf         = zeros(WINDOW_SIZE, N_FEATURES);
buf_count   = 0;
chirp_count = 0;

fprintf('MATLAB radar running — 5 class fusion mode...\n');

%% ================= MAIN LOOP =================
while true
    t0 = tic;

    %% ---- Receive ----reali
    raw = rx();
    raw = double(raw(:));
    if numel(raw) < actual_n
        raw = [raw; zeros(actual_n - numel(raw), 1)];
    elseif numel(raw) > actual_n
        raw = raw(1:actual_n);
    end

    %% ---- Dechirp & Profiles ----
    dechirped     = raw .* conj(ref_chirp);
    range_profile = abs(fft(dechirped, nfft));
    range_profile = range_profile(1:max_range_bin);

    magnitude  = abs(raw);
    phase      = angle(raw);
    phase_diff = diff(phase);

    f = zeros(1, N_FEATURES);

    %% ---- BLOCK 1: 20 ML Model Features ----
    f(1)  = mean(magnitude);
    f(2)  = std(magnitude);
    f(3)  = max(magnitude);
    f(4)  = mean(phase);
    f(5)  = std(phase);
    f(6)  = sum(magnitude.^2);                      % energy

    [f(12), idx] = max(range_profile);              % peak_power, range bin
    f(7)  = idx;
    f(8)  = mean(range_profile);
    f(9)  = std(range_profile);
    f(10) = mean(phase_diff);
    f(11) = std(phase_diff);
    f(13) = median(magnitude);

    mag_sorted = sort(magnitude);
    n_s = length(mag_sorted);
    f(14) = mag_sorted(max(1, round(n_s*0.25)));
    f(15) = mag_sorted(min(n_s, round(n_s*0.75)));

    mag_sum = sum(magnitude);
    if mag_sum > 0
        mag_norm = magnitude / mag_sum;
        mag_norm(mag_norm < 1e-10) = 1e-10;
        f(16) = -sum(mag_norm .* log2(mag_norm));
    end

    f(17) = sum(abs(diff(sign(real(raw)))) > 0) / 2;
    f(18) = sqrt(mean(magnitude.^2));
    f(19) = f(3) / (f(1) + 1e-10);
    if f(2) > 0
        f(20) = mean(((magnitude - f(1)) / f(2)).^4);
    end

    %% ---- BLOCK 2: 10 Physics Features ----

    % Doppler spectrum (one-sided, normalized)
    doppler_spec      = abs(fft(dechirped, nfft));
    doppler_half      = doppler_spec(1:floor(nfft/2));
    ds_norm           = doppler_half / (sum(doppler_half) + 1e-10);
    freq_bins         = (0:length(ds_norm)-1);

    % F21: Doppler centroid
    f(21) = sum(freq_bins .* ds_norm') / (sum(ds_norm) + 1e-10);

    % F22: Doppler bandwidth (weighted std)
    f(22) = sqrt(sum(((freq_bins - f(21)).^2) .* ds_norm') / (sum(ds_norm) + 1e-10));

    % F23: Doppler symmetry (positive vs negative energy ratio)
    pos_e = sum(doppler_spec(2:floor(nfft/2)));
    neg_e = sum(doppler_spec(floor(nfft/2)+1:end));
    f(23) = pos_e / (neg_e + 1e-10);

    % F24: Number of Doppler peaks (limb count proxy)
    [~, dp] = findpeaks(ds_norm, 'MinPeakProminence', 0.005);
    f(24) = length(dp);

    % F25: Range entropy
    rp_norm = range_profile / (sum(range_profile) + 1e-10);
    rp_norm(rp_norm < 1e-10) = 1e-10;
    f(25) = -sum(rp_norm .* log2(rp_norm));

    % F26: Trunk/limb energy ratio
    trunk_cut = max(1, round(length(ds_norm) * 0.05));
    f(26) = sum(ds_norm(1:trunk_cut)) / (sum(ds_norm(trunk_cut+1:end)) + 1e-10);

    % F27-28: Cadence from autocorrelation
    ac     = xcorr(magnitude - mean(magnitude), 'normalized');
    ac_pos = ac(ceil(end/2)+1:end);
    [ac_pks, ac_locs] = findpeaks(ac_pos, 'MinPeakProminence', 0.05);
    if ~isempty(ac_pks)
        f(27) = ac_locs(1);    % cadence period in samples
        f(28) = ac_pks(1);     % cadence strength (0-1)
    else
        f(27) = 0;
        f(28) = 0;
    end

    % F29: Instantaneous frequency std
    inst_freq = diff(unwrap(phase));
    f(29) = std(inst_freq);

    % F30: SNR estimate (dB)
    sig_pwr   = mean(magnitude.^2);
    noise_pwr = mean(mag_sorted(1:max(1, round(n_s*0.1))).^2) + 1e-10;
    f(30) = 10 * log10(sig_pwr / noise_pwr);

    %% ---- Sliding Window Buffer ----
    buf         = circshift(buf, -1, 1);
    buf(end,:)  = f;
    buf_count   = min(buf_count + 1, WINDOW_SIZE);
    chirp_count = chirp_count + 1;

    %% ---- Write CSV with flag ----
    if buf_count == WINDOW_SIZE && mod(chirp_count, WRITE_EVERY) == 0

        if isfile(FLAG_FILE), delete(FLAG_FILE); end

        fid = fopen(CSV_FILE, 'w');
        fprintf(fid, header);
        for r = 1:WINDOW_SIZE
            fprintf(fid, [repmat('%.6f,', 1, N_FEATURES-1), '%.6f\n'], buf(r,:));
        end
        fclose(fid);

        fid2 = fopen(FLAG_FILE, 'w');
        fprintf(fid2, 'ready');
        fclose(fid2);

        fprintf('Chirp %d | DoppBW=%.1f | Cadence=%.0f | TrunkRatio=%.2f | SNR=%.1fdB\n', ...
                chirp_count, f(22), f(27), f(26), f(30));
    end

    elapsed = toc(t0);
    if elapsed < 0.05, pause(0.05 - elapsed); end
end
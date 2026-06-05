clc; clear; close all;

%% ================= CONSTANTS =================
c = 3e8;

%% ================= FMCW PARAMETERS =================
B  = 56e6;
Tm = 3e-3;
fs = 1e6;
master_clk = 40e6;
fc = 2.4e9;
Ns = round(Tm*fs);
slope = B/Tm;

fprintf('========== FMCW RECEIVER ==========\n');
fprintf('Bandwidth: %.1f MHz\n', B/1e6);
fprintf('Range resolution: %.2f cm\n', c/(2*B)*100);
fprintf('Samples/chirp: %d\n', Ns);
fprintf('Max Range Limit: 0.5 m\n');
fprintf('===================================\n\n');

%% ================= REFERENCE CHIRP =================
t = (0:Ns-1).' / fs;
ref_chirp = exp(1j*pi*slope*t.^2);

%% ================= USRP RX =================
fprintf('Initializing USRP B210...\n');

rx = comm.SDRuReceiver( ...
    'Platform', 'B210', ...
    'SerialNum', '31CF3B0', ...
    'CenterFrequency', fc, ...
    'Gain', 60, ...
    'MasterClockRate', master_clk, ...
    'DecimationFactor', master_clk/fs, ...
    'SamplesPerFrame', Ns, ...
    'OutputDataType', 'double', ...
    'ChannelMapping', 1);

fprintf('✅ USRP initialized\n\n');

%% ================= SYNC =================
fprintf('Syncing with TX');
for k = 1:50
    rx();
    if mod(k,10)==0, fprintf('.'); end
end
fprintf(' Done!\n\n');

%% ================= RANGE AXIS =================
N_half = floor(Ns/2);
k = (0:N_half-1).';
rangeAxis = (c*k*Tm)/(2*B);

% CLEAN ROOM CONFIGURATION
MIN_R = 0.40;   % 40 cm
MAX_R = 0.50;   % 50 cm (MAX RANGE REQUIRED)
WALL_R = 0.50;  % Ignore beyond 50 cm

zone = (rangeAxis >= MIN_R) & (rangeAxis <= MAX_R);
ranges_zone = rangeAxis(zone);

fprintf('=========================================\n');
fprintf('CLEAN ROOM MODE\n');
fprintf('Detection zone: %.0fcm - %.0fcm\n', MIN_R*100, MAX_R*100);
fprintf('TX-RX distance expected: 50-60cm\n');
fprintf('Detection bins: %d\n', sum(zone));
fprintf('=========================================\n\n');

%% ================= CALIBRATION =================
fprintf('EMPTY ROOM CALIBRATION\n');
fprintf('Make sure no one is between 40-50cm\n');
pause(5);

Ncal = 100;
baseline = zeros(sum(zone), Ncal);

for i = 1:Ncal
    [rxSig, len] = rx();
    if len < Ns, continue; end
    
    beat = rxSig .* conj(ref_chirp);
    beat = beat - mean(beat);
    R = fft(beat .* hamming(Ns));
    baseline(:, i) = abs(R(zone)).^2;
end

furnitureBaseline = mean(baseline, 2);
noiseStd = std(baseline, 0, 2);

fprintf('Calibration Complete\n\n');
pause(2);

%% ================= DETECTION =================
numChirps = 32;
prevCube = [];
detectionHistory = zeros(1,6);
historyIdx = 1;
occupied = false;

figure('Name','FMCW Occupancy Clean Room','Position',[50 50 1400 800]);

frame = 0;

try
    while true
        
        cube = zeros(sum(zone), numChirps);
        validChirps = 0;
        
        for m = 1:numChirps
            [rxSig, len] = rx();
            if len < Ns, continue; end
            
            beat = rxSig .* conj(ref_chirp);
            beat = beat - mean(beat);
            R = fft(beat .* hamming(Ns));
            cube(:, m) = abs(R(zone)).^2;
            validChirps = validChirps + 1;
        end
        
        if validChirps < numChirps/2
            continue;
        end
        
        %% ENERGY
        energy = mean(cube,2);
        deltaE = abs(energy - furnitureBaseline);
        energyThresh = 1.8 * noiseStd;   % relaxed
        energyBins = sum(deltaE > energyThresh);
        energy_detected = energyBins >= 2;
        
        %% MOTION
        motion_detected = false;
        motionMetric = 0;
        motionThreshold = 0.4 * mean(noiseStd);  % more sensitive
        
        if ~isempty(prevCube)
            frameDiff = abs(cube - prevCube);
            motionMetric = mean(frameDiff(:));
            motion_detected = motionMetric > motionThreshold;
        end
        prevCube = cube;
        
        %% DOPPLER VARIANCE
        doppler_var = var(cube,0,2);
        doppler_mean = mean(doppler_var);
        highVarBins = sum(doppler_var > 2*doppler_mean);
        doppler_detected = highVarBins >= 2;
        
        %% PEAK
        [peakEnergy, peakIdx] = max(energy);
        peakRange = ranges_zone(peakIdx);
        baselinePeak = furnitureBaseline(peakIdx);
        strongPeak = (peakEnergy > 1.6*baselinePeak) && (peakRange <= WALL_R);
        
        %% SCORE
        score = 0;
        if energy_detected, score = score + 2; end
        if motion_detected, score = score + 3; end
        if doppler_detected, score = score + 1; end
        if strongPeak, score = score + 2; end
        
        currentDetection = score >= 3;

        
        %% HISTORY
        detectionHistory(historyIdx) = currentDetection;
        historyIdx = mod(historyIdx,6) + 1;
        %occupied = sum(detectionHistory) >= 3;
        if ~occupied
    % Harder to turn ON
    if sum(detectionHistory) >= 4
        occupied = true;
    end
else
    % Easier to turn OFF
    if sum(detectionHistory) <= 1
        occupied = false;
    end
end

        
        frame = frame + 1;
        
        %% PRINT STATUS
        fprintf('[F%03d] ', frame);
        if occupied
            if motion_detected
                fprintf('🚶 MOVING @%.2fm\n', peakRange);
            else
                fprintf('🧍 STATIC @%.2fm\n', peakRange);
            end
        else
            fprintf('⬜ EMPTY\n');
        end
        
        %% VISUALIZATION
        subplot(2,3,1);
        imagesc(1:numChirps, ranges_zone, 10*log10(cube+eps));
        axis xy; colormap jet; colorbar;
        xlabel('Chirp'); ylabel('Range (m)');
        title('Range-Time Map');
        xline(numChirps,'k');
        caxis([-60 -20]);
        
        subplot(2,3,2);
        plot(ranges_zone,10*log10(energy),'b','LineWidth',2); hold on;
        plot(ranges_zone,10*log10(furnitureBaseline),'k--');
        plot(ranges_zone(peakIdx),10*log10(peakEnergy),'ro','LineWidth',2);
        xline(WALL_R,'r--','LineWidth',2);
        hold off;
        xlim([MIN_R MAX_R]);
        title(sprintf('Range Profile (%.2fm)',peakRange));
        grid on;
        
        subplot(2,3,3);
        plot(ranges_zone,deltaE,'r','LineWidth',2); hold on;
        plot(ranges_zone,energyThresh,'k--');
        hold off;
        xlim([MIN_R MAX_R]);
        title('Energy Change');
        grid on;
        
        subplot(2,3,4);
        plot(ranges_zone,doppler_var,'m','LineWidth',2); hold on;
        yline(2*doppler_mean,'r--');
        hold off;
        xlim([MIN_R MAX_R]);
        title('Doppler Variance');
        grid on;
        
        subplot(2,3,5);
        bar([energy_detected*2, motion_detected*3, doppler_detected, strongPeak*2]);
        title(sprintf('Score = %d',score));
        ylim([0 3.5]);
        grid on;
       subplot(2,3,6);
plot(1:6, detectionHistory, 'bo-', ...
    'LineWidth', 2, ...
    'MarkerSize', 10, ...
    'MarkerFaceColor', 'b');
hold on;

if occupied
    fill([0 7 7 0], [-0.2 -0.2 1.2 1.2], ...
        'g', 'FaceAlpha', 0.25, 'EdgeColor', 'none');
    text(3.5, 0.6, 'OCCUPIED', ...
        'FontSize', 16, ...
        'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center');
else
    fill([0 7 7 0], [-0.2 -0.2 1.2 1.2], ...
        'r', 'FaceAlpha', 0.25, 'EdgeColor', 'none');
    text(3.5, 0.6, 'EMPTY', ...
        'FontSize', 16, ...
        'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center');
end

yline(0.5, 'k--', 'LineWidth', 1.5);
hold off;

xlabel('Frame History');
ylabel('Detection');
title(sprintf('History (%d/6, need 3)', sum(detectionHistory)));

ylim([-0.2 1.2]);
xlim([0 7]);
grid on;

drawnow;
pause(0.05);

        
    end
    
catch ME
    release(rx);
    fprintf('\nStopped after %d frames\n', frame);
    if ~strcmp(ME.identifier,'MATLAB:interruption')
        fprintf('Error: %s\n', ME.message);
    end
end

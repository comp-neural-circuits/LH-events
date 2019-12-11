% Main script
%
% Title: Adaptation of spontaneous activity in the developing visual
% cortex
% Authors: Marina E. Wosniack, Jan H. Kirchner, Ling-Ya Chao, Nawal
% Zabouri, Christian Lohmann, Julijana Gjorgjieva
% Submitted: December 2019
%
% Marina E. Wosniack and Ling-Ya Chao
% marina-elaine.wosniack@brain.mpg.de
% 


clear; close all;

%saving simulation details: True or false
saving_info = true;

% dynamics type: cov, adapt
% cov: Hebbian input-threshold learning rule
% adapt: Hebbian input-threshold learning rule + adapted H-events

type = 'adapt';

% number of cells
N_in = 50; N_out = 50;

% initial weight and bias
W_initial = [0.15 0.25];

bias = 0.05;
spread = 4;

% time resolution
total_ms = 5000;
dt_per_ms = 1000;

% time constants (in ms)
tau_w = 500;
tau_out = 0.01;
tau_theta = 1;

% thresholds
corr_thres = 0.6;
W_thres = [0.0 0.5];
bounded = true;

% parameters for events
L_dur = 0.15;
H_dur = 0.15;
L_p = 1.5;
H_p = 4.5;
% if no H-events should be present in the simulation, set H_p = Inf
L_pct = [0.2 0.8];
H_pct = [0.8 1.0];
H_amp = 6;
L_amp = 1;

% file naming
folder_name = sprintf('../results/%s', datestr(now, 'mmmdd'));
if ~exist(folder_name, 'dir')
    mkdir(folder_name)
end

subfolder_name = sprintf(['%s/%s_%s'], ...
    folder_name, datestr(now, 'HHMM'), type);
if ~exist(subfolder_name, 'dir')
    mkdir(subfolder_name)
end

if saving_info == true
    weights_name = sprintf('%s/weights.mat', subfolder_name);
    weights_sparse_name = sprintf('%s/weights_sparse.mat', subfolder_name);
    output_name = sprintf('%s/output.mat', subfolder_name);
    theta_name = sprintf('%s/theta.mat', subfolder_name);
end

eventlog = fopen(sprintf('%s/eventlog.txt', subfolder_name), 'w');
paramslog = fopen(sprintf('%s/params.txt', subfolder_name), 'w');

% run simulation
[spars, wid, align, topo, record_W, record_times, plot_W, plot_times, ...
    record_output, record_theta] = ...
    independent_rates( ...
    type, N_in, N_out, ...
    W_initial, bias, spread, ...
    total_ms, dt_per_ms, ...
    W_thres, bounded, corr_thres, ...
    L_p, H_p, L_dur, H_dur, L_pct, H_pct, L_amp, H_amp, ...
    tau_w, tau_out, tau_theta, ...
    subfolder_name, eventlog);

% saving the parameters used in the results folder
fprintf(paramslog, 'type_%s\n bias_%.2f\nLd%.2f\nHd%.2f\nLp%.2f\nHp%.2f\nHamp%.2f\nTw%.2f\nTout%.2f\nTtheta%.2f\nWthr%.2f\ncorrthr%.2f', ...
    type, bias, ...
    L_dur, H_dur, ...
    L_p, H_p, ...
    H_amp, ...
    tau_w, tau_out, tau_theta, ...
    W_thres(2), corr_thres);

% save data
if saving_info == true
    save(weights_name, 'record_W', 'record_times', '-v7.3');
    save(weights_sparse_name, 'plot_W', 'plot_times');
    save(output_name, 'record_output', 'record_times');
    save(theta_name, 'record_theta', 'record_times');
end

fclose(eventlog);
fclose(paramslog);

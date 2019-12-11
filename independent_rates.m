% Weights evolution function
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


function [sparsity, width, align, topo, record_W, record_times_ms, ...
    plot_W, plot_times_ms, record_output, record_theta] = ...
    independent_rates( ...
    type, N_in, N_out, ...
    W_initial, bias, spread, ...
    total_ms, dt_per_ms, ...
    W_thres, bounded, corr_thres, ...
    L_p, H_p, L_dur, H_dur, L_pct, H_pct, L_amp, H_amp, ...
    tau_w, tau_out, tau_theta, ...
    subfolder_name, eventlog)

switch type
    case 'cov'
        type_id = 0;
    case 'adapt'
        type_id = 1;
    otherwise
        warning('unexpected type');
end

dt = 1 / dt_per_ms;

% initialize weights
W = biased_weights(N_in, W_initial, bias, spread);

% initialize activities
in = zeros(N_in, 1);
out = zeros(N_out, 1);
out_spon = zeros(N_out, 1);
theta = zeros(N_out, 1);

% initialize counters
L_counter = round(exprnd(L_p * dt_per_ms)) + 1;
L_dur_counter = 0;

H_counter = isinf(H_p) * (-1) + ...
    ~isinf(H_p) * (round(dt_per_ms * gamrnd(H_p, 1)) + 1);
H_dur_counter = 0;

record_counter = 1;
plot_counter = 1;

% DENSE RECORD
% initialize matrices for recording weights, output, theta
record_freq = 1;
fprintf(eventlog, 'record freq: every %.2f ms \n', record_freq);

record_times_ms = [0 : record_freq : (total_ms)];
record_times_dt = int32(record_times_ms * dt_per_ms);
num_of_records = length(record_times_dt);

record_W = zeros(N_out, N_in, num_of_records);
record_W(:,:,1) = W;

record_output = zeros(N_out, num_of_records);

record_output(:,1) = out;

record_theta = zeros(N_out, num_of_records);
record_theta(:,1) = theta;

% SPARSE RECORD
% initialize matrices for summary plotting
plot_W_freq = 1000;

plot_times_ms = 0 : plot_W_freq : total_ms;
num_of_plots = length(plot_times_ms);

plot_W = zeros(N_out, N_in, num_of_plots);
plot_W(:,:,1) = W;

% starting dynamics
for t = 1 : total_ms * dt_per_ms
    
    if L_counter == 0
        L_length = round(N_in * (L_pct(1) + rand(1) * (L_pct(2) - L_pct(1))));
        L_start = randsample(N_in, 1);
        
        in = zeros(N_in, 1);
        in(mod(L_start : L_start + L_length - 1, N_in) + 1) = L_amp;
        
        L_dur_counter = round(normrnd(L_dur, L_dur * 0.1) * dt_per_ms);
        rand_per_L = round(exprnd(L_p * dt_per_ms));
        L_counter = rand_per_L + L_dur_counter;
        
        center = mod(L_start + round(L_length / 2), N_in);
        fprintf(eventlog, '%d L %d %d \n', t, center, L_length);
    end
    
    if H_counter == 0
        H_length = round(N_out * (H_pct(1) + rand(1) * (H_pct(2) - H_pct(1))));
        H_start = randsample(N_out, 1);
        
        out_spon = zeros(N_out, 1);
        out_spon(mod(H_start : H_start + H_length - 1, N_out) + 1) = normrnd(H_amp, H_amp / 3, H_length, 1);
        
        if type_id == 1 % adapt
            out_spon = out_spon .* theta;
        end
        
        H_dur_counter = round(normrnd(H_dur, H_dur * 0.1) * dt_per_ms);
        rand_per_H = round(dt_per_ms * gamrnd(H_p, 1));
        H_counter = rand_per_H + H_dur_counter;
        
        center = mod(H_start + round(H_length / 2), N_in);
        fprintf(eventlog, '%d H %d %d \n', t, center, H_length);
    end
    
    % output vector
    out = out + (dt / tau_out) * (- out + out_spon + W * in);
    
    % different LRs
    switch type_id
        case 0 % cov
            dW = (dt / tau_w) * out * (in - corr_thres)';
            
        case 1 % adapt
            dW = (dt / tau_w) * out * (in - corr_thres)';
            theta = theta + (dt / tau_theta) * (-theta + out);
    end
    
    % update weight matrix
    
    W = W + dW;
    
    if bounded
        W(W < W_thres(1)) = W_thres(1);
        W(W > W_thres(2)) = W_thres(2);
    end
    
    % counter operations
    L_counter = L_counter - 1;
    H_counter = H_counter - 1;
    L_dur_counter = L_dur_counter - 1;
    H_dur_counter = H_dur_counter - 1;
    
    % end of events
    if L_dur_counter == 0
        in = zeros(N_in, 1);
    end
    if H_dur_counter == 0
        out_spon = zeros(N_out, 1);
    end
    
    % densely record W, output, theta
    if mod(t, record_freq * dt_per_ms) == 0 && ismember(t, record_times_dt)
        record_counter = record_counter + 1;
        %saving the weights...
        record_W(:,  :, record_counter) = W;
        %the output activity...
        record_output(:, record_counter) = out;
        %and theta...
        record_theta(:, record_counter) = theta;
    end
    
    % sparsely record W
    if mod(t, plot_W_freq * dt_per_ms) == 0
        plot_counter = plot_counter + 1;
        fprintf('completion %.2f %% \n', t / (total_ms * dt_per_ms) * 100);
        
        plot_W(:, :, plot_counter) = W;
        
        if all(all(round(W, 2) == W_thres(1))) || ...
                all(all(round(W, 2) == W_thres(2)))
            fprintf('termination: all lit or all died \n');
            break;
        end
    end
end

% Finding which weights constitute the receptive field at the end of the simulation
aux_matrix = zeros(50,50);
for ii = 1:50
    for jj = 1:50
        if(W(ii,jj) > W_thres(1) + (max(plot_W(:)) - W_thres(1)) / 5)
            aux_matrix(ii,jj) = 1;
        end
    end
end

% Finding the center of each receptive field
left = zeros(50,1);
right = zeros(50,1);
center = zeros(50,1);
for jj = 1:N_in
    found = 0;
    i = 1;
    while ((i < N_in)) && (found == 0)
        left(i) = sum(aux_matrix(jj, (mod(i - (N_in/2 - 1) - 1:i - 1, N_in) + 1)));
        right(i) = sum(aux_matrix(jj, (mod(i - 1:(i + N_in/2 - 1) - 1, N_in) + 1)));
        if abs(left(i) - right(i)) < 3 && aux_matrix(jj,i) > 0
            center(jj) = i;
            found = 1;
        end
        i = i+1;
    end
end


%calculate the distance from the diagonal
distance_diagonal = zeros(1,50);
total = N_in;
for jj = 1:N_in
    if(sum(aux_matrix(jj , :)) < 1)
        distance_diagonal(jj) = 0;
        total = total - 1;
    else
        distance_diagonal(jj) = min(mod(center(jj) - jj, N_in),mod(jj - center(jj), N_in));
    end
end

diag_error = 1 / total * sum(distance_diagonal.^2);
active = W > W_thres(1) + (max(plot_W(:)) - W_thres(1)) / 5;
bar_width = sum(active, 2);
avg_width = mean(bar_width(bar_width > 5));
spars = sum(bar_width > 5) / N_out;

%the statistics
column_rf_error = (N_in / 2) * N_in / 6;
sparsity = 1 - spars;
width = avg_width / N_in;
align = 1 - diag_error / column_rf_error;
topo = align * (1 - sparsity)^2;

%FIRST FIGURE - final receptive field

%colormap
molaspass=interp1([1 51 102 153 204 256], [0 0 0; 0 0 .75; .5 0 .8; 1 .1 0; 1 .9 0; 1 1 1], 1:256);

h1 = figure;
imagesc(W);
colormap(molaspass);
set(gca, 'FontSize', 16);
yticks([1 25 50])
xticks([1 25 50])
xlabel('thalamus')
ylabel('cortex')
colorbar; caxis([W_thres(1) max(plot_W(:))]);
name_rf = sprintf('%s/matrix_rule_%d_theta_%.2f_H_p_%.2f.png', subfolder_name, type_id, corr_thres,H_p);
set(h1, 'Units', 'Inches');
pos = get(h1, 'Position');
set(h1, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
print(h1, name_rf, '-dpng', '-r500');


%SECOND FIGURE - synapses to cortical cell

h4 = figure;
title('Synapses to a cortical cell')
subplot(2,2,1);
extracted = reshape(double(plot_W(10,:,:)) / 1, [N_in, num_of_plots]);
plot(plot_times_ms', extracted');
xlabel('Time'); ylabel('W(10,i)');
set(gca, 'FontSize', 10);
xlim([0 max(plot_times_ms(:))])
subplot(2,2,2);
extracted = reshape(double(plot_W(20,:,:)) / 1, [N_in, num_of_plots]);
plot(plot_times_ms', extracted');
set(gca, 'FontSize', 10);
xlim([0 max(plot_times_ms(:))])
xlabel('Time'); ylabel('W(20,i)');
subplot(2,2,3);
extracted = reshape(double(plot_W(30,:,:)) / 1, [N_in, num_of_plots]);
plot(plot_times_ms', extracted');
set(gca, 'FontSize', 10);
xlim([0 max(plot_times_ms(:))])
xlabel('Time'); ylabel('W(30,i)');
subplot(2,2,4);
extracted = reshape(double(plot_W(40,:,:)) / 1, [N_in, num_of_plots]);
plot(plot_times_ms', extracted');
set(gca, 'FontSize', 10);
xlim([0 max(plot_times_ms(:))])
xlabel('Time'); ylabel('W(40,i)');
set(h4,'Units','Inches');
pos = get(h4,'Position');
set(h4, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
name_syn_to_cort = sprintf('%s/syntocortical', subfolder_name);
print(h4,name_syn_to_cort,'-dpng', '-r500');

%THIRD FIGURE - synapses from a thalamic cell

h5 = figure;set(h5, 'Visible', 'off');
title('Synapses from a retinal cell')
subplot(2,2,1);
extracted = reshape(double(plot_W(:,10,:)) / 1, [N_in, num_of_plots]);
plot(plot_times_ms', extracted');
xlabel('Time'); ylabel('W(j,10)');
set(gca, 'FontSize', 10);
xlim([0 max(plot_times_ms(:))])
subplot(2,2,2);
extracted = reshape(double(plot_W(:,20,:)) / 1, [N_in, num_of_plots]);
plot(plot_times_ms', extracted');
set(gca, 'FontSize', 10);
xlim([0 max(plot_times_ms(:))])
xlabel('Time'); ylabel('W(j,20)');
subplot(2,2,3);
extracted = reshape(double(plot_W(:,30,:)) / 1, [N_in, num_of_plots]);
plot(plot_times_ms', extracted');
set(gca, 'FontSize', 10);
xlim([0 max(plot_times_ms(:))])
xlabel('Time'); ylabel('W(j,30)');
subplot(2,2,4);
extracted = reshape(double(plot_W(:,40,:)) / 1, [N_in, num_of_plots]);
plot(plot_times_ms', extracted');
set(gca, 'FontSize', 10);
xlim([0 max(plot_times_ms(:))])
xlabel('Time'); ylabel('W(j,40)');
set(h5,'Units','Inches');
pos = get(h5,'Position');
set(h5,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
name_syn_from_thal = sprintf('%s/synfromthalamical', subfolder_name);
print(h5, name_syn_from_thal, '-dpng', '-r500');
end


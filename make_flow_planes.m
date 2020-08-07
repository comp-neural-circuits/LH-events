%
% Adaptation of spontaneous activity in the developing visual cortex
% M. E. Wosniack et al.
% 
% Codes to plot the flow planes
% Auxiliar functions file: 
%
% Author: Marina Wosniack
% Max Planck Institute for Brain Research
% marina-elaine.wosniack@brain.mpg.de
% June 2020
%
% This code calculates the flow plane of the weight dynamics.
%
% The system is linear, and we solve it by diagonalizing the covariance 
% matrix of inputs.
%
% First, it reduces the dimensionality of the dynamics to 2, by grouping
% synaptic weights into two categories: potentiating and depressing 
% weights.
% 
% The size of the potentiating group (w+) is given by the size of the 
% desired receptive field (rf_size variable). The remaining weights (w-) 
% will be grouped as depressing.
%
% What we study in the phase plane is the evolution of the average 
% potentiating and depressing groups.
%
% The weight evolution follows the Hebbian rule, where changes in the 
% weights are proportional to post_syn_act * (pre_syn_act - theta_u).
%
% Since our inputs are well-structured, we calculate exactly its
% correlation matrix and solve the linear system (see the paper).
%
% This code works for scenarios with and without H-events (just set it
% to zero with Hamp = 0). 
%
% The code returns a figure with the flow plane, a sample trajectory 
% with initial conditions that match those of our simulations, and the
% fixed point of the dynamics.
%

clear; close all;
rng(1); % fixing a seed

%===================================================================%
% Code parameters 

S = 50; % number of neurons in each layer of the network
rf_size = 23; % desired receptive field size outcome
%
% Properties of initial weight matrix; same as in simulations
W_ini = [0.15 0.25];
bias = 0.05;
spread = 4;
%
% Event sizes
l_ini = 0.2;
l_end = 0.8;
h_ini = 0.8;
h_end = 1;
%
% Are there H-events? (If not, Hamp == 0)
H_amp = 0;
% Average size of input (L-event)
avg_size_L = (l_end + l_ini) / 2;
%
% This defines all the inputs to the covariance matrix.
l_interval = (l_ini * S : 0.02 * S : l_end * S);
% Not used
h_interval = (h_ini * S : 0.02 * S : h_end * S);
% Length of the evolution of initial conditions [in s]
total_len = 1;
total_len_traj = total_len / 5; % we simulate trajectories for a shorter period
tau_w = 500; % Hebbian rule time constant
%
% This defines the grid of the phase plane.. not that with H-events,
% since the fixed point moves, it is necessary to adjust the intervals of
% the grid
[weight_in_x, weight_in_y] = meshgrid(-0.21 : 0.05 : 0.51,-0.21 : 0.05 : 0.51);
%
%===================================================================%


%%
% This creates the super matrix with all the possible inputs
% We assume here that they are all equiprobable
super_mat = [];
for jj = 1 : size(l_interval, 2)
    s = l_interval(jj);
    t_PN = zeros(S, S);
    ini = 0;
    for l = 1:S
        for m = 1:s
            kk = mod(m + ini, S) + 1;
            t_PN(l, kk) = 1;
        end
        ini = ini + 1;
    end
    super_mat = [super_mat; t_PN];
end
% And now we calculate the input correlation matrix
Corr = zeros(S, S);
for ii = 1:S
    for jj = 1:S
        Corr(ii, jj) = mean(super_mat(:, ii).*super_mat(:, jj));
    end
end
%%
% Computing now the covariance matrix, which we will diagonalize
% Notice that the receptive field size will now be transformed into 
% an input threshold value
%
% First we find the last element along the diagonal that will be part 
% of the desired receptive field, which sits along the diagonal
next_cond = Corr(1, ceil(rf_size / 2));
% And now transform this into the theta_u value
theta_u = next_cond / (avg_size_L);
% Now calculating the covariance matrix and diagonalizing it
C = cov(super_mat);
Cov = C + avg_size_L * (avg_size_L - theta_u);
% Diagonalizing it and making it sure that the weights that potentiate
% are the ones along the diagonal
for ee = 1:40
    [V,D] = eig(Cov);
    [D,I] = sort(diag(D), 'descend');
    V = V(:, I);
    
    % Now V1 = V(:,1) and V2 = V(:,2) already have information about who 
    % potentiates and who depresses
    %
    w_tendency = biased_weights(S, W_ini, bias + 1, spread);
    e1 = w_tendency * V(:,1);
    e2 = w_tendency * V(:,2);
    
    pot1 = e1 * V(:,1)' + e2 * V(:,2)' >= 0;
    if(pot1(S/2, S/2) == 1)
        break;
    end
end

% This is another trick to define w+ and w-

mat_aux = zeros(S, S);
for hh = 1:S
    vec = zeros(1, S);
    vec(mod(hh - floor(rf_size / 2)-1 : hh + floor(rf_size / 2) - 1, S) + 1) = 1;
    mat_aux(hh, :) = vec;
end
pot1 = logical(mat_aux);
dep1 = ~pot1;
%
mat_aux = zeros(S, S);
for hh = 1:S
    vec = zeros(1, S);
    vec(mod(hh + 25 - floor(rf_size / 2) - 1 : hh + 25 + floor(rf_size / 2) - 1, S) + 1) = 1;
    mat_aux(hh,:) = vec;
end
pot2 = logical(mat_aux);
dep2 = ~pot2;

%%
% Now doing the evolution of each initial conditions
% 
% How many initial conditions do I want?
ini_pos = zeros(size(weight_in_x,2));
ini_neg = zeros(size(weight_in_x,2));
fin_pos = zeros(size(weight_in_x,2));
fin_neg = zeros(size(weight_in_x,2));
%
ll = 1; % this counter will go over each arrow
for gg = 1 : size(weight_in_x, 2) % going over x-coordinates in the plane
    for nn = 1 : size(weight_in_y, 2) % going over y-coordinates now
        final_weight = zeros(S, S);
        ini_weight = zeros(S, S);
        %
        W_pos = weight_in_x(gg, nn) * ones(S, S);
        W_neg = weight_in_y(gg, nn) * ones(S, S);
        %
        if(weight_in_x(gg, nn) < weight_in_y(gg, nn))
            pot = dep2;
            dep = pot2;
        else
            pot = pot1;
            dep = dep1;
        end
        
        W_pos = pot.*W_pos;
        mean_pos = nanmean(W_pos(W_pos ~= 0));
        W_neg = dep.*W_neg;
        mean_neg = nanmean(W_neg(W_neg ~= 0));
        
        % If H-events, delta > 0 
        delta = - theta_u * H_amp * ones(1, S);
        W = W_pos + W_neg;
        ini_pos(gg, nn) = mean_pos;
        ini_neg(gg, nn) = mean_neg;
        ini_weight(:, :, ll) = W;
        c = zeros(S, S);
        d = zeros(S, S);
        w = zeros(S, S, 1);
        delta2 = zeros(S, S);
        %
        for jj = 1:S
            for ii = 1:S
                delta2(ii,jj) = delta * V(:,ii);
                c(ii,jj) = W(jj,:) * V(:,ii);
                d(ii,jj) = c(ii,jj) + delta2(ii,jj) / D(ii);
            end
        end
        %
        for jj = 1:S
            sum_w = 0;
            for ii = 1:S
                sum_w = sum_w + (d(ii, jj) * exp(D(ii) * total_len / tau_w) - delta2(ii, jj) / D(ii)) * V(:, ii);
            end
            w(:, jj, 1) = sum_w;
        end
        %
        up = w; up(dep) = NaN;
        down = w; down(pot) = NaN;
        fin_pos(gg, nn) = nanmean(nanmean(up, 2));
        fin_neg(gg, nn) = nanmean(nanmean(down, 2));
    end
    ll = ll + 1;
end
%%
% Finally the figure
figure;
% First we normalize the arrows.. otherwise they will have diff sizes and 
% we just want to show the direction, not the magnitude
norm = sqrt((fin_pos - ini_pos).^2 + (fin_neg - ini_neg).^2);
quiver(ini_pos, ini_neg, (fin_pos - ini_pos)./norm,(fin_neg - ini_neg)./norm, 0.4, 'linewidth', 0.5)
%
xlabel('W_+');
ylabel('W_-');
axis('square');
hold on;
% now we calculate the trajectory of the biased initial condition on top
w_ini = biased_weights(S, W_ini, bias, spread);
ini_x = mean(w_ini(1, mod(1 - floor(rf_size / 2) : 1 + floor(rf_size / 2), S) + 1));
ini_y = mean(w_ini(1, 1 + floor(rf_size / 2) : S));
%
coords = make_trajectory(S, ini_x, ini_y, V, D, pot1, dep1, ...
    theta_u, H_amp, total_len_traj, tau_w);
plot(coords(:, 1), coords(:, 2), 'r-');
% and finally add the location of the fixed point
bvec = H_amp * theta_u * ones(1, size(Cov, 2));
fp = linsolve(Cov, bvec.');
plot(mean(fp), mean(fp),'go');
name_fig = strcat('figures/flow_theta_', num2str(theta_u), '_H_', num2str(H_amp), '.pdf');
% just aesthetics
set(findall(gca, 'Type', 'Line'),'LineWidth',2);
set(gcf,'renderer','painters')
set(0,'DefaultAxesFontWeight', 'normal', ...
    'DefaultAxesFontSize', 16, ...
    'DefaultAxesFontAngle', 'normal', ... 
    'DefaultAxesFontWeight', 'normal', ...
    'DefaultAxesTitleFontWeight', 'normal', ...
    'DefaultAxesTitleFontSizeMultiplier', 1);
%
xlim([0 0.5]);
ylim([0 0.5]);
xticks([0 0.25 0.5]);
yticks([0 0.25 0.5]);
print(name_fig,'-dpdf','-r500');
%
clear;
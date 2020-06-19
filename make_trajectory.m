function traj_coords = make_trajectory(S, ini_x, ini_y, V, D, pot, dep, theta_u, H_amp, total_len2, tau_w)
% This function computes the trajectory of an initial condition {ini_x,
% ini_y} in the flow plane until the bounds are reached.
%
% 

% Parameters:
% just a trick to make the trajectories stop in the bounds
MMAX = 0.4999995;
MMAX2 = 0.5;
MMIN = 0.00001;
%
traj_coords = [];
while((ini_x >= MMIN || ini_y >= MMIN) && (abs(ini_x) < MMAX && abs(ini_y)< MMAX))
    W_pos = ini_x * ones(S, S);
    W_neg = ini_y * ones(S, S);
    W_pos = pot.*W_pos;
    mean_pos = nanmean(W_pos(W_pos ~= 0));
    W_neg = dep.*W_neg;
    mean_neg = nanmean(W_neg(W_neg ~= 0));
    delta = - theta_u * H_amp * ones(1, S); 
    W = W_pos + W_neg;
    ini_pos2 = mean_pos;
    ini_neg2 = mean_neg;
    c = zeros(S, S);
    d = zeros(S, S);
    w = zeros(S, S);
    delta2 = zeros(S, S);
    for jj = 1:S
        for ii = 1:S
            delta2(ii, jj) = delta * V(:, ii);
            c(ii, jj) = W(jj, :)*V(:, ii);
            d(ii, jj) = c(ii, jj) + delta2(ii, jj) / D(ii);
        end
    end
    %
    for jj = 1 : S
        sum_w = 0;
        for ii = 1 : S
            sum_w = sum_w + (d(ii, jj) * exp(D(ii) * total_len2 / tau_w) - delta2(ii, jj) / D(ii)) * V(:, ii);
        end
        w(:, jj) = sum_w;
    end
    %
    up = w; up(dep) = NaN;
    down = w; down(pot) = NaN;
    ini_x = nanmean(nanmean(up, 2));
    ini_y = nanmean(nanmean(down, 2));
    %
    if ini_x < 0 || ini_x > MMAX2
        ini_x = ini_pos2;
    end
    %
    if ini_y < 0 || ini_y > MMAX2
        ini_y = ini_neg2;
    end
    %
   if (ini_x == ini_pos2 && ini_y == ini_neg2)
       break;
   end
    traj_coords = [traj_coords; [ini_pos2, ini_neg2]];
end
% This function generates a biased initial distribution of
% weights
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

function W = biased_weights(N_in, W_initial, bias, spread)
    W = zeros(N_in);

    for i = 0 : floor(N_in / 2)
        for j = 1 : N_in
            W(j, mod(j + i - 1, N_in) + 1) = normpdf(i, 0, spread);
            W(j, mod(j - i - 1, N_in) + 1) = normpdf(i, 0, spread);
        end
    end

    W = W / normpdf(0, 0, spread);
    
    W = bias * W + W_initial(1) + (W_initial(2) - W_initial(1)) * rand(N_in);
    W(W < 0) = 0;
end





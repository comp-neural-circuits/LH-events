%% Figure 7 Panel B + Supplementary
% Generates scatter plot of H-event amplitude vs. average preceding activity. Also generates plot that illustrates the effect of varying the inclusion criterion.
% 
% Title: Adaptation of spontaneous activity in the developing visual
% cortex
% Authors: Marina E. Wosniack, Jan H. Kirchner, Ling-Ya Chao, Nawal
% Zabouri, Christian Lohmann, Julijana Gjorgjieva
% Submitted: December 2019
%
% Jan H Kirchner
% jan.kirchner@brain.mpg.de
%



%first need to load: Friederike_data.mat which contains the variable data
close all
%load Friederike_data.mat
load('matlab.mat');
data = (EXCELWITHANIMALID_CP); data = data(: , 1:12); 
%%
% select only events satisfying the criterion from (Siegel et al 2012)
low_thr = 20; %percent of low cut off
high_thr = 80; %percent of L vs H events cut off
INTWINDOW = 100; %size of window in seconds
lowerBound = 10; % minimal amount of L-H-pairs required to be included in analysis

cc = doRegression(data , low_thr , high_thr , INTWINDOW , lowerBound , 1);
%%
windowSpace = linspace(10 , 200 , 10);
ccs = []; ccs5 = []; ccs95 = [];
for xx = 1:length(windowSpace)
    xx
    cc = doRegression(data , low_thr , high_thr , floor(windowSpace(xx)) , lowerBound , 0);
    ccs = [ccs ; mean(cc)];
    ccs5 = [ccs5 ; prctile(cc,5)];
    ccs95 = [ccs95 ; prctile(cc,95)];
end
%%
figure; hold on
p1 = plot(windowSpace , ccs , 'Color' , rgb('red') , 'LineWidth' , 1);
p2 = plot(windowSpace , ccs5 , 'Color' , rgb('black') , 'LineWidth' , 1 , 'LineStyle' , '--');
plot(windowSpace , ccs95 , 'Color' , rgb('black') , 'LineWidth' , 1 , 'LineStyle' , '--');
axis square;
xlabel('size of integration window (sec)')
ylabel('correlation coefficient')
legend([p1 , p2] , {'mean' , 'sem (bootstrap)'} , 'Location' , 'southeast');
legend boxoff;
xlim([0 , 200]); xticks([0 , 100 , 200])
ylim([-0.75 , 0.75]); yticks(-0.75:0.25:0.75)

%%
boundSpace = 5:1:25;
ccs = []; ccs5 = []; ccs95 = []; cAnimals = [];
for xx = 1:length(boundSpace)
    xx
    [cc,countAnimals] = doRegression(data , low_thr , high_thr , INTWINDOW , boundSpace(xx) , 0);
    ccs = [ccs ; mean(cc)];
    ccs5 = [ccs5 ; prctile(cc,5)];
    ccs95 = [ccs95 ; prctile(cc,95)];
    cAnimals = [cAnimals ; countAnimals];
end
%%
figure; hold on
p1 = plot(boundSpace , ccs , 'Color' , rgb('red') , 'LineWidth' , 1);
p2 = plot(boundSpace , ccs5 , 'Color' , rgb('black') , 'LineWidth' , 1 , 'LineStyle' , '--');
plot(boundSpace , ccs95 , 'Color' , rgb('black') , 'LineWidth' , 1 , 'LineStyle' , '--');
axis square;
xlabel('lower bound criterion for inclusion of events')
ylabel('correlation coefficient')
legend([p1 , p2] , {'mean' , 'sem (bootstrap)'} , 'Location' , 'southeast');
legend boxoff;
xlim([5 , 25]); xticks(5:5:25)
ylim([-0.75 , 0.75]); yticks(-0.75:0.25:0.75)
%%
figure;
plot(boundSpace , cAnimals , 'Color' , rgb('blue') , 'LineWidth' , 1 );
ylabel('number of animals')
xlabel('lower bound criterion for inclusion of events')
xlim([5 , 25]); xticks(5:5:25)

%%

function [cc , countAnimals] = doRegression(data , low_thr , high_thr , INTWINDOW , lowerBound , plotFlag)
    PRS = table2array(data(:,7)); % get participation rates
    %%
    Hi = find(PRS>=high_thr); %indices of the H events in the large array with all data
    Li = find(PRS>=low_thr & PRS<high_thr); %indices of the L events in the large array with all data
    %%
    Ha=table2array(data(Hi,6)); % get H event amplitudes

    start_frames = table2array(data(:,2)); % get start frames of events

    ds = diff(start_frames); % get time between
    % need to identify break points where new recording starts
    new_in = find(ds<0) + 1; new_in = [1; new_in];

    %initialize accumulator arrays
    mean_vec = zeros(size(Hi)); x = zeros(size(Hi)); 
    anIDS = zeros(length(Hi) , 1);
    [anID , ~] = grp2idx(categorical(table2array(data(: , 11))));
    % iterate over all H-events
    for ii=1:length(Hi)
        % get local integration window and identify candidate events
        LOCALINTWINDOW = floor(INTWINDOW/data.factortomultiplyframeby(Hi(ii)));
        CANDIDS = find(data.Startframe < data.Startframe(Hi(ii)) - LOCALINTWINDOW);
        CANDIDS = CANDIDS(CANDIDS < Hi(ii)); CANDIDS = max(CANDIDS);

        % select only those preceding events that occured within the same
        % recording
        d = Hi(ii) - new_in;
        d2 = find(d>0);
        x(ii) = max(d2);

        if ii>1 & ~isempty(CANDIDS)
            %make sure it only counts the L events since the previous H event
            since_index = max(new_in(x(ii)), CANDIDS);      
        else
            %unless it's the first H event
            since_index = new_in(x(ii));
        end
        %take all L events preceding that H event 
        allL_vec = since_index:Hi(ii)-1; 

        L_vec = intersect(allL_vec,[Li ; Hi]); % get all events (L and H) that preceded the current event
        mean_vec(ii) = mean(table2array(data(L_vec,6))); %compute mean amplitude
        anIDS(ii) = anID(Hi(ii)); % accumulate animal ID
    end
    %%
    % combine data per animal and compute number of L-H pairs
    uAnIDS = unique(anIDS); % unique animal IDs
    manyFLAG = []; accDX = []; accDY = []; % initialize accumulators
    countAnimals = 0;
    for xx = 1:length(uAnIDS)
        dx = mean_vec(anIDS == uAnIDS(xx)); dy = Ha(anIDS == uAnIDS(xx)); % select only events belonging to this animal
        useID = ~isnan(dx) & ~isnan(dy); % exclude nans
        manyFLAG = [manyFLAG ; ones(length(dx) , 1)* length(dx(useID))]; % construct array with number of events per animal
        accDX = [accDX ; dx]; accDY = [accDY ; dy];  % accumulate L-H pairs
        if length(dx) > lowerBound
            countAnimals = countAnimals + 1;
        end
    end

    %%
    dx = accDX(manyFLAG > lowerBound); dy = accDY(manyFLAG > lowerBound);
    useID = ~isnan(dx) & ~isnan(dy);
    [RGL , LB , UB , DX , cc] = bootstrapRegression(dx(useID),dy(useID) , 100);
    if plotFlag
        figure; hold on;
        scatter( dx , dy , 50 , 'MarkerFaceColor' ,  rgb('lightgrey') , 'MarkerEdgeColor' , rgb('black'));
        plot(DX , RGL , 'LineWidth' , 2 , 'Color' , rgb('red'))
        plot(DX , LB , 'LineWidth' , 1 , 'Color' , rgb('black') , 'LineStyle' , '--')
        plot(DX , UB , 'LineWidth' , 1 , 'Color' , rgb('black') , 'LineStyle' , '--')
        xlabel('Average preceding activity (mean F/F_0)')
        ylabel('H-event amplitude (mean F/F_0)')
        xlim([0.9 , 1.6]); ylim([0.9 , 1.6])
        axis square
    end
end

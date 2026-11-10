%% load data
% first store the DFA,PSD and rho values from .csv to .mat

% load("window_data_ST_sw.mat")

%% Criticality indicators vs p_base
% Matrix dimensions: [num_pbase × num_repeat]

p = p_base_values(:);

data = {dfa, -1*PSD, rho};

ylabels = {
    '\alpha_{DFA}'
    '-PSD slope'
    '\rho(M)'
};

% Threshold ranges
thresholds = {
    [0.9, 1.1]
    [0.9,1.1]
    [0.95, 1.05]
};

colors = [
    101/256 55/256 172/256;
    201/256 117/256 149/256;
    248/256 220/256 132/256;
    ];

figure('Color', 'w', 'Position', [100 100 750 950]);

% t = tiledlayout(3, 1, ...
%     'TileSpacing', 'compact', ...
%     'Padding', 'compact');

hold on
% hold on
for i = 1:3

    % ax = nexttile;
    % hold(ax, 'on');

    % Mean and standard deviation
    y_mean = mean(data{i}, 2, 'omitnan');
    y_std  = std(data{i}, 0, 2, 'omitnan');

    % Sort by p_base
    [p_sorted, idx] = sort(p);

    y_mean = y_mean(idx);
    y_std  = y_std(idx);

    % Shaded standard deviation
    fill( ...
        [p_sorted; flipud(p_sorted)], ...
        [y_mean-y_std; flipud(y_mean+y_std)], ...
        colors(i,:), ...
        'FaceAlpha', 0.18, ...
        'EdgeColor', 'none', ...
        'HandleVisibility', 'off');

    % Mean curve
    plot(p_sorted, y_mean, ...
        'Color', colors(i,:), ...
        'LineWidth', 2, ...
        'DisplayName', 'Mean \pm SD');

    % Horizontal threshold reference lines
    th = thresholds{i};

    for j = 1:2
        yline(th(j), ':', ...
            'Color', [0.55 0.55 0.55], ...
            'LineWidth', 1, ...
            'HandleVisibility', 'off');
    end

    % Find p_base at threshold crossings
    for j = 1:2

        p_cross = find_threshold_crossings( ...
            p_sorted, y_mean, th(j));

        % Draw vertical dashed lines
        for k = 1:length(p_cross)

            xline(p_cross(k), '--', ...
                'Color', colors(i,:), ...
                'LineWidth', 1.3, ...
                'HandleVisibility', 'off');
        end

        fprintf('%s: threshold = %.2f, p_base = %s\n', ...
           th(j), mat2str(p_cross, 4));

    end

    ylabel(ylabels{i}, 'FontSize', 13);
    

    xlim([min(p), max(p)]);
    grid on;
    box on;

    set(gca, ...
        'FontSize', 11, ...
        'LineWidth', 1, ...
        'TickDir', 'out');

end

% xlabel(t, 'p_{base}', 'FontSize', 14);

% Export
exportgraphics(gcf, ...
    'criticality_indicators.png', ...
    'Resolution', 300);

ylim([0,1.2])
grid off

% %% for 1 panel figure
xlabel('p_{base}')
ylabel('Data')
legend DFA -PSD \rho
grid off
%% Bootstrap uncertainty of joint critical window

p = p_base_values(:);

% Indicators: [num_pbase x num_repeat]
data = {dfa, -PSD, rho};

thresholds = [
    0.90, 1.10;
    0.90, 1.10;
    0.95, 1.05
];

n_bootstrap = 2000;

rng(42);

n_p = numel(p);
n_repeat = size(dfa, 2);

% Check dimensions
assert(isequal(size(dfa), size(PSD), size(rho)), ...
    'All indicators must have identical dimensions.');

assert(size(dfa, 1) == n_p, ...
    'Number of rows must match p_base_values.');

% Sort p and all indicator matrices
[p_sorted, idx] = sort(p);

for k = 1:3
    data{k} = data{k}(idx, :);
end

%% Original window: mean curves

mean_curves = zeros(n_p, 3);

for k = 1:3
    mean_curves(:, k) = mean(data{k}, 2, 'omitnan');
end

[W_original, intervals_original] = ...
    calculate_joint_window(p_sorted, mean_curves, thresholds);

fprintf('Original joint window width: %.6f\n', W_original);

disp('Original critical intervals:');
disp(intervals_original);


%% Bootstrap

W_boot = zeros(n_bootstrap, 1);

% Set this according to your simulation design:
% true: repeat r uses a matched realization across different p
% false: simulations at different p are independent
paired_across_p = false;

for b = 1:n_bootstrap

    boot_curves = zeros(n_p, 3);

    if paired_across_p
        % Same repeat indices across all p
        selected = randi(n_repeat, [1, n_repeat]);
    end

    for i = 1:n_p

        if ~paired_across_p
            selected = randi(n_repeat, [1, n_repeat]);
        end

        % Same repeat indices for DFA, PSD, rho
        for k = 1:3
            boot_curves(i, k) = ...
                mean(data{k}(i, selected), 'omitnan');
        end
    end

    [W_boot(b), ~] = ...
        calculate_joint_window(p_sorted, boot_curves, thresholds);
end


%% Uncertainty

CI_width = prctile(W_boot, [2.5, 97.5]);

W_boot_mean = mean(W_boot);
W_boot_std = std(W_boot);

fprintf('\n===== Critical window uncertainty =====\n');

fprintf('Original width     = %.6f\n', W_original);
fprintf('Bootstrap mean     = %.6f\n', W_boot_mean);
fprintf('Bootstrap SD       = %.6f\n', W_boot_std);

fprintf('95%% bootstrap CI   = [%.6f, %.6f]\n', ...
    CI_width(1), CI_width(2));


%% Plot bootstrap distribution

figure('Color', 'w');

histogram(W_boot, 40, 'Normalization', 'pdf');

hold on;

xline(W_original, 'r-', 'LineWidth', 2);

xline(CI_width(1), 'k--', 'LineWidth', 1.5);
xline(CI_width(2), 'k--', 'LineWidth', 1.5);

xlabel('Critical window width');
ylabel('Bootstrap density');

title('Bootstrap uncertainty of joint critical window');

box on;

%% Local function: threshold crossings
function p_cross = find_threshold_crossings(p, y, threshold)

    p = p(:);
    y = y(:);

    p_cross = [];

    for k = 1:length(p)-1

        y1 = y(k);
        y2 = y(k+1);

        if ~isfinite(y1) || ~isfinite(y2)
            continue;
        end

        % Exact match
        if y1 == threshold
            p_cross(end+1) = p(k);
        end

        % Crossing between adjacent points
        if (y1-threshold)*(y2-threshold) < 0

            p_interp = p(k) + ...
                (threshold-y1) * ...
                (p(k+1)-p(k)) / (y2-y1);

            p_cross(end+1) = p_interp;

        end

    end

    if y(end) == threshold
        p_cross(end+1) = p(end);
    end

    p_cross = unique(p_cross);

end

function [W, intervals] = calculate_joint_window(p, curves, thresholds)

% p:          [num_p x 1]
% curves:     [num_p x 3]
% thresholds: [3 x 2]
%
% W: total width of all intervals satisfying
%    the three simultaneous threshold conditions.
%
% intervals: [num_intervals x 2]

p = p(:);

if any(~isfinite(curves(:))) || any(~isfinite(p))
    W = NaN;
    intervals = zeros(0, 2);
    return;
end

if any(diff(p) <= 0)
    error('p must be strictly increasing.');
end

% Start from original sampled p values
breakpoints = p;

% Add all threshold crossing locations
for k = 1:3

    for j = 1:2

        crossing = find_threshold_crossings( ...
            p, curves(:, k), thresholds(k, j));

        breakpoints = [breakpoints; crossing(:)];

    end
end

breakpoints = unique(sort(breakpoints));

intervals = zeros(0, 2);

% Between adjacent breakpoints, threshold membership
% cannot change under piecewise-linear interpolation.

for i = 1:numel(breakpoints)-1

    left = breakpoints(i);
    right = breakpoints(i+1);

    mid = (left + right)/2;

    values = zeros(1, 3);

    for k = 1:3
        values(k) = interp1(p, curves(:, k), mid, 'linear');
    end

    inside = all( ...
        values > thresholds(:, 1)' & ...
        values < thresholds(:, 2)' ...
        );

    if inside
        intervals(end+1, :) = [left, right];
    end
end

% Merge adjacent intervals
if ~isempty(intervals)

    merged = intervals(1, :);

    for i = 2:size(intervals, 1)

        if abs(intervals(i,1)-merged(end,2)) < 1e-12

            merged(end,2) = intervals(i,2);

        else

            merged(end+1,:) = intervals(i,:);

        end
    end

    intervals = merged;
end

% Total joint window width
W = sum(intervals(:,2)-intervals(:,1));

end
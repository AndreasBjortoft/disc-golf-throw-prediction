clc; clear; close all;

% Load the CSV file
data = readtable("C:\Users\TE18B\Downloads\train_data1.csv");  % Ensure correct path

% Extract the targets (dependent variable) and variables (independent variables)
y = data.Speed;     % Assuming 'Speed' is the dependent variable
exclude_cols = [];  % Example: Excluding columns 3 and 5
X = data{:, setdiff(2:7, exclude_cols)};  % Exclude selected columns


% Set parameters
num_samples = 1000;  % Number of samples
step_size = 0.05;    % Reduced step size for stability
width = 1.0;         % Slice width

% Perform slice sampling
beta_samples = slice_sampling(y, X, num_samples, step_size, width);


% Plot trace plots for all beta coefficients
figure;
num_vars = size(beta_samples, 2);  % Number of variables (beta coefficients)
num_rows = ceil(num_vars / 2);      % Number of rows for subplots
num_cols = 2;                       % 2 columns for the subplot grid

for j = 1:num_vars
    subplot(num_rows, num_cols, j);  % Adjust subplot layout dynamically
    plot(1:num_samples, beta_samples(:, j), 'LineWidth', 1);
    title(['Trace plot for \beta_', num2str(j)]);
    xlabel('Iteration');
    ylabel(['\beta_', num2str(j)]);
end

% Posterior means and 95% credible intervals
posterior_means = mean(beta_samples);  % Mean of each beta
credible_intervals = quantile(beta_samples, [0.025, 0.975]);  % 95% CI

disp('Posterior Means:');
disp(posterior_means);
disp('95% Credible Intervals:');
disp(credible_intervals);

% Plot posterior distributions


burn_in = 400;  % Set the burn-in period (e.g., first 200 iterations)
beta_samples_post = beta_samples(burn_in+1:end, :);  % Remove burn-in samples

figure;
for j = 1:num_vars
    subplot(num_rows, num_cols, j);
    histogram(beta_samples_post(:, j), 30, 'FaceColor', 'b', 'EdgeColor', 'k');
    title(['Posterior distribution of \beta_', num2str(j)]);
    xlabel(['\beta_', num2str(j)]);
    ylabel('Frequency');
end

% Define the CSV file path
csvFile = "C:\Users\TE18B\Downloads\test_data1.csv";  % Replace with your actual file path

% Load data from CSV file (assuming the first row contains headers)
data = readtable(csvFile);

% Extract dependent variable (actual disc speed)
actual_speeds = data{:, 1};  % First column is actual disc speed

% Extract independent variables (X data)
X = data{:, setdiff(2:size(data, 2), exclude_cols)};  % Exclude same columns

  % Columns 2 to 6 are independent variables

% Posterior mean values for beta coefficients
beta_mean = posterior_means;

% Compute predicted disc speed for each row
predicted_speeds = X * beta_mean';

% Compute Mean Squared Error (MSE)
errors = actual_speeds - predicted_speeds;
mse = sqrt(mean(errors.^2));

% Display actual vs. predicted values
fprintf('Actual Speed | Predicted Speed\n');
fprintf('------------------------------\n');
for i = 1:length(actual_speeds)
    fprintf('%12.4f | %16.4f\n', actual_speeds(i), predicted_speeds(i));
end

% Print Mean Squared Error
fprintf('\nMean Squared Error (MSE): %.4f\n', mse);

% --- Visualization of Actual vs Predicted Speeds ---
figure;

% Scatter plot of actual vs predicted speeds
scatter(actual_speeds, predicted_speeds, 'filled', 'MarkerFaceColor', 'b');
hold on;

% Add diagonal line (y=x) to represent perfect prediction
plot([min(actual_speeds), max(actual_speeds)], [min(actual_speeds), max(actual_speeds)], 'r--');  % Diagonal line (y=x)

% Labeling
xlabel('Actual Speed');
ylabel('Predicted Speed');
title('Actual vs Predicted Speed');
axis equal;  % Ensure equal scaling on both axes
grid on;

% Slice Sampling Function
function [beta_samples] = slice_sampling(y, X, num_samples, step_size, width)
    % Number of observations and variables
    [n, p] = size(X);

    % Initialize beta (random start can be used too)
    beta = zeros(p, 1);
    
    % Store samples
    beta_samples = zeros(num_samples, p);

    % Define log-likelihood (Gaussian likelihood)
    log_likelihood = @(beta) -0.5 * sum((y - X*beta).^2);

    % Define log-prior (Normal(0, 10^2) prior)
    log_prior = @(beta) -0.5 * sum((beta / 10).^2);
    
    % Slice sampling loop
    for i = 1:num_samples
        for j = 1:p
            % Set up the slice
            beta_j = beta(j);
            log_posterior = log_likelihood(beta) + log_prior(beta);
            u = log_posterior + log(rand);  % Random level for the slice
            
            % Initialize slice bounds
            lower_bound = beta_j - width;
            upper_bound = beta_j + width;
            
            % Perform slice sampling step
            while true
                proposed_beta_j = lower_bound + (upper_bound - lower_bound) * rand;
                beta_temp = beta;
                beta_temp(j) = proposed_beta_j;
                proposed_log_posterior = log_likelihood(beta_temp) + log_prior(beta_temp);

                if proposed_log_posterior > u
                    beta(j) = proposed_beta_j;
                    break;
                else
                    % Shrink the bounds
                    if proposed_beta_j > beta_j
                        upper_bound = proposed_beta_j;
                    else
                        lower_bound = proposed_beta_j;
                    end
                end
            end
        end
        
        % Store the current sample
        beta_samples(i, :) = beta';
    end
end
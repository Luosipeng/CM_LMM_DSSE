function generate_case33bw_timeseries(output_path, n_steps, seed, sample_minutes, matpower_root)
%GENERATE_CASE33BW_TIMESERIES Generate sensor measurements by MATPOWER AC PF.
%
% Mathematical model
% ------------------
% At time t, each base load is perturbed by
%   P_i(t) = P_i^0 * g(t) * exp(0.08 z_i(t)),
%   z(t)   = rho z(t-1) + sqrt(1-rho^2) L epsilon(t),
% where g(t) combines daily harmonics and a system-wide AR(1) component,
% and LL' is an exponential covariance based on electrical graph distance.
% For every t, MATPOWER solves the nonlinear AC power-flow equations
%   S_i = V_i * conj(sum_j Y_ij V_j).
% Thus all exported sensor values are samples of an actual solved case33bw
% operating point, rather than values synthesized directly in Python.

arguments
    output_path (1,:) char
    n_steps (1,1) double {mustBeInteger,mustBePositive} = 2016
    seed (1,1) double {mustBeInteger} = 2026
    sample_minutes (1,1) double {mustBePositive} = 15
    matpower_root (1,:) char = 'D:\luosipeng\matpower8.1'
end

addpath(matpower_root);
addpath(fullfile(matpower_root, 'data'));
define_constants;
rng(seed, 'twister');

mpc0 = loadcase('case33bw');
base_pd = mpc0.bus(:, PD);
base_qd = mpc0.bus(:, QD);
sensor_buses = [2 6 9 13 18 22 25 29 31 33];
feature_names = {'vm_pu', 'va_degree', 'p_inj_mw', 'q_inj_mvar', ...
                 'p_upstream_mw', 'q_upstream_mvar'};
n_bus = size(mpc0.bus, 1);
n_sensor = numel(sensor_buses);
n_feature = numel(feature_names);

% Graph distance produces spatially correlated local load innovations.
active = mpc0.branch(:, BR_STATUS) > 0;
graph_obj = graph(mpc0.branch(active, F_BUS), mpc0.branch(active, T_BUS));
distance = distances(graph_obj);
spatial_cov = exp(-distance / 6) + 1e-6 * eye(n_bus);
spatial_chol = chol(spatial_cov, 'lower');
branch_impedance = hypot(mpc0.branch(active, BR_R), mpc0.branch(active, BR_X));
electrical_graph = graph(mpc0.branch(active, F_BUS), ...
                         mpc0.branch(active, T_BUS), branch_impedance);
sensor_electrical_distance = distances(electrical_graph, sensor_buses, sensor_buses);

% Reduce the radial 33-bus feeder to directed edges between sensor nodes.
% A sensor's parent is its nearest upstream sensor on the path to slack bus 1.
sensor_edges = zeros(0, 2);
tree_obj = minspantree(graph_obj, 'Root', 1);
for child_idx = 1:n_sensor
    child_bus = sensor_buses(child_idx);
    path_to_root = shortestpath(tree_obj, 1, child_bus);
    upstream = path_to_root(ismember(path_to_root, sensor_buses));
    upstream(upstream == child_bus) = [];
    if ~isempty(upstream)
        parent_bus = upstream(end);
        parent_idx = find(sensor_buses == parent_bus, 1);
        sensor_edges(end + 1, :) = [parent_idx child_idx]; %#ok<AGROW>
    end
end

sensor_values = nan(n_steps, n_sensor, n_feature);
full_bus_vm = nan(n_steps, n_bus);
full_bus_va = nan(n_steps, n_bus);
load_scale = nan(n_steps, n_bus);
converged = false(n_steps, 1);
local_state = zeros(n_bus, 1);
global_state = 0;
rho_local = 0.92;
rho_global = 0.97;
mpopt = mpoption('verbose', 0, 'out.all', 0, 'pf.alg', 'NR');

for t = 1:n_steps
    hour = mod((t - 1) * sample_minutes / 60, 24);
    daily = 0.82 + 0.16 * sin(2 * pi * (hour - 8) / 24) ...
                  + 0.06 * sin(4 * pi * (hour - 17) / 24);
    global_state = rho_global * global_state + sqrt(1-rho_global^2) * randn;
    local_state = rho_local * local_state ...
        + sqrt(1-rho_local^2) * spatial_chol * randn(n_bus, 1);
    scale = max(0.35, daily + 0.035 * global_state) .* exp(0.08 * local_state);
    load_scale(t, :) = scale';

    mpc = mpc0;
    mpc.bus(:, PD) = base_pd .* scale;
    % Small power-factor movement is temporally and spatially coupled.
    q_scale = scale .* max(0.8, 1 + 0.025 * local_state);
    mpc.bus(:, QD) = base_qd .* q_scale;
    result = runpf(mpc, mpopt);
    converged(t) = result.success;
    if ~result.success
        continue;
    end

    full_bus_vm(t, :) = result.bus(:, VM)';
    full_bus_va(t, :) = result.bus(:, VA)';
    for s = 1:n_sensor
        bus_id = sensor_buses(s);
        incoming = find(result.branch(:, T_BUS) == bus_id & ...
                        result.branch(:, BR_STATUS) > 0, 1);
        if isempty(incoming)
            p_up = 0;
            q_up = 0;
        else
            p_up = result.branch(incoming, PT);
            q_up = result.branch(incoming, QT);
        end
        sensor_values(t, s, :) = [result.bus(bus_id, VM), ...
            result.bus(bus_id, VA), -result.bus(bus_id, PD), ...
            -result.bus(bus_id, QD), p_up, q_up];
    end
end

if ~all(converged)
    warning('%d/%d power-flow samples did not converge.', sum(~converged), n_steps);
end

time_minutes = (0:n_steps-1)' * sample_minutes;
base_bus_pd = base_pd;
base_bus_qd = base_qd;
case_name = 'case33bw';
base_mva = mpc0.baseMVA;
output_dir = fileparts(output_path);
if ~isempty(output_dir) && ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
save(output_path, 'sensor_values', 'sensor_buses', 'sensor_edges', ...
    'sensor_electrical_distance', 'feature_names', 'full_bus_vm', ...
    'full_bus_va', 'load_scale', ...
    'converged', 'time_minutes', 'base_bus_pd', 'base_bus_qd', ...
    'base_mva', 'case_name', 'sample_minutes', 'seed', '-v7');
fprintf('Saved %d converged case33bw samples to %s\n', sum(converged), output_path);
end

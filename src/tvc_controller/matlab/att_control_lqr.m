clear; clc;

%% Parameters (example)
T  = 120;      % thrust [N]
l  = 0.25;     % lever arm [m]
Ix = 0.70;     % inertia [kg*m^2]
Iy = 0.80;
Iz = 0.60;

bx = T*l/Ix;   % p_dot coefficient for delta_y
by = T*l/Iy;   % q_dot coefficient for delta_x
bz = 1/Iz;     % r_dot coefficient for tau_z

%% State: x = [phi theta psi p q r]'
A = [0 0 0 1 0 0;
     0 0 0 0 1 0;
     0 0 0 0 0 1;
     0 0 0 0 0 0;
     0 0 0 0 0 0;
     0 0 0 0 0 0];

% Input: u = [delta_x delta_y tau_z]'
B = [0   0   0;
     0   0   0;
     0   0   0;
     0   bx  0;   % p_dot = bx * delta_y
     by  0   0;   % q_dot = by * delta_x
     0   0   bz]; % r_dot = bz * tau_z

%% LQR weights
% Penalize angles heavily, rates moderately
Q = diag([300 300 120,  30 30 20]);
% Penalize gimbal angles and yaw torque (tune these!)
R = diag([1.0 1.0 0.2]);

K = lqr(A,B,Q,R);  % u = -K x
disp("LQR gain K = "); disp(K);

%% Simple simulation (Euler integration + saturation)
dt = 0.001;
t  = 0:dt:6;

% Initial condition: roll=8deg, pitch=-6deg, yaw=15deg
x = zeros(6, numel(t));
x(:,1) = [deg2rad(200); deg2rad(0); deg2rad(0); 0; 0; 0];

% Actuator limits
delta_max = deg2rad(10);    % gimbal angle limit
tauz_max  = 2.0;           % yaw torque limit [N*m]

u = zeros(3, numel(t));

for k = 1:numel(t)-1
    u_cmd = -K * x(:,k);

    % saturations
    u(1,k) = min(max(u_cmd(1), -delta_max), delta_max); % delta_x
    u(2,k) = min(max(u_cmd(2), -delta_max), delta_max); % delta_y
    u(3,k) = min(max(u_cmd(3), -tauz_max),  tauz_max);  % tau_z

    xdot = A*x(:,k) + B*u(:,k);
    x(:,k+1) = x(:,k) + dt*xdot;
end
u(:,end) = u(:,end-1);

%% Plots (2 subplots in one figure + autoscale with margins)

% ---------- helper: set axis limits with margins ----------
autoMargin = @(v, m) deal( ...
    (min(v) - m * max(eps, max(v) - min(v))), ...
    (max(v) + m * max(eps, max(v) - min(v))) );

m = 0.10;   % 10% margin
tmin = min(t); tmax = max(t);

figure('Name','LQR Attitude & Inputs','Color','w');

% ===================== Subplot 1: Attitude =====================
ax1 = subplot(2,1,1); grid(ax1,'on'); hold(ax1,'on'); box(ax1,'on');

y_att = rad2deg(x(1:3,:));
plot(ax1, t, y_att(1,:), 'LineWidth', 1.5);
plot(ax1, t, y_att(2,:), 'LineWidth', 1.5);
plot(ax1, t, y_att(3,:), 'LineWidth', 1.5);

xlabel(ax1,'Time [s]');
ylabel(ax1,'Angle [deg]');
legend(ax1,'\phi roll','\theta pitch','\psi yaw','Location','best');
title(ax1,'Attitude response (LQR)');

xlim(ax1, [tmin tmax]);
[y1min, y1max] = autoMargin(y_att(:), m);
ylim(ax1, [y1min y1max]);

set(ax1, 'FontSize', 12, 'LineWidth', 1.2);

% ===================== Subplot 2: Inputs (yyaxis) =====================
ax2 = subplot(2,1,2); grid(ax2,'on'); hold(ax2,'on'); box(ax2,'on');

% Left axis: gimbal angles
yyaxis(ax2,'left');
u_gim = rad2deg(u(1:2,:));
plot(ax2, t, u_gim(1,:), 'LineWidth', 1.5);
plot(ax2, t, u_gim(2,:), 'LineWidth', 1.5);
ylabel(ax2,'Gimbal [deg]');

% Right axis: yaw torque
yyaxis(ax2,'right');
plot(ax2, t, u(3,:), 'LineWidth', 1.5);
ylabel(ax2,'\tau_z [N·m]');

xlabel(ax2,'Time [s]');
title(ax2,'Control inputs (with saturation)');
legend(ax2, '\delta_x','\delta_y','\tau_z','Location','best');

xlim(ax2, [tmin tmax]);

% Apply margins to both y-axes separately (important for yyaxis)
yyaxis(ax2,'left');
[yLmin, yLmax] = autoMargin(u_gim(:), m);
ylim(ax2, [yLmin yLmax]);

yyaxis(ax2,'right');
[yRmin, yRmax] = autoMargin(u(3,:), m);
ylim(ax2, [yRmin yRmax]);

set(ax2, 'FontSize', 12, 'LineWidth', 1.2);

% Optional: tighten vertical spacing a bit
% (MATLAB R2019b+ has tiledlayout which is even nicer; ask if you want that)


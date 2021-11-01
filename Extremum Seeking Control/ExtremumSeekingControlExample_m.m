clc; clear all; close all
s = tf('s');

T = 1500;
Ts = 0.1; % Sampling time [s]
x_off = 5;
y_off = -10;

D_w = 0.4; % Dither frequency [rad/s]
D_A = 1; % Dither aplimtude [-]

M_w = D_w; % Demodulation frequency [rad/s]
M_A = 2/D_A; % Demodulation amplitude [-]
M_phi = 0/180*pi; % Demodulation phase offset [rad]

w_A = 0.5; % Disturbance amplitude

Ki = 0.0025; % Integrator gain

F.wcLPF = 0.23; % Cut-off frequency low-pass filter [rad/s]
F.wcHPF = 0.205; % Cut-in frequency high-pass filter [rad/s]
F.HPF = s/(s+F.wcHPF);
F.LPF = F.wcLPF/(s+F.wcLPF);

a = sim('ExtremumSeekingControlExample', T);
b = a.get('simout');
assignin('base','b',b);
simout = b;
clear a b

%% Plot result

figure(1)
P.w = logspace(-2, 1, 200);
[P.HPFmag, P.HPFphase] = magphase(F.HPF, P.w);
[P.LPFmag, P.LPFphase] = magphase(F.LPF, P.w);
mbode(P.HPFmag, P.HPFphase, P.w, [], [], [], 'b', [], true)
mbode(P.LPFmag, P.LPFphase, P.w, [], [], [], 'k', [], true)
vline(D_w), subplot(211), vline(D_w)
sgtitle('ESC - Filter and frequency design', 'Interpreter', 'latex') 
legend('High-pass filter', 'Low-pass filter')

P.t = simout.Time;
P.u = simout.Data(:,1);
P.K = simout.Data(:,2);
P.y = simout.Data(:,3);
P.N = length(P.t);

figure(2)
plot(P.t, P.u, 'LineWidth', 2, 'Color', 'k'), hold on
plot(P.t, P.y, 'LineWidth', 2, 'Color', 'b')
plot(P.t, P.K, 'LineWidth', 1, 'Color', 'r', 'LineStyle', '--')
plot(P.t, ones(1, P.N).*x_off, '-.k', 'LineWidth', 1, 'Color', 0.3*ones(1,3))
plot(P.t, ones(1, P.N).*y_off, '-.k', 'LineWidth', 1, 'Color', 0.3*ones(1,3))
legend('Input', 'Integrator state', 'Output', 'Location', 'SouthEast', 'Interpreter', 'latex')
grid on
xlabel('Time [s]', 'Interpreter', 'latex')
ylabel('Value [-]', 'Interpreter', 'latex')
title('Extremum Seeking Control optimization', 'Interpreter', 'latex')
set(gca,'TickLabelInterpreter','latex');
clc, clear all, close all

% Simulation parameters
Ts = 0.01;
TMax = 1000;
simmdl = 'WTTSRTracker_UncertaintyStability';

% Load global and turbine constants
loadglobalconstants()
T = loadturbineconstants('NREL5MW');

% Initial values
Tg0 = 2.4e4;
Wr0 = 9*rpm2rads;
V0 = 9;

% Load prior information on rotor performance (Cp, Ct, Cq)
load(['TablesCpCtCmCq.mat'])
T.Cp = Tables.Cp';
T.Ct = Tables.Ct';
T.CTSR = Tables.TSR;
T.CPitch = Tables.Pitch*deg2rad;

[NDTSR, NDPITCH] = ndgrid(Tables.TSR, Tables.Pitch);
T.CPfunc = griddedInterpolant(NDTSR, NDPITCH, Tables.Cp');
T.CTfunc = griddedInterpolant(NDTSR, NDPITCH, Tables.Ct');

% Add Cq table
TSRMesh = repmat(T.CTSR,[length(T.CPitch) 1])';
T.Cq = T.Cp./TSRMesh;

% Find maximum Cp and corresponding TSR
T.CpMax = max(max(T.Cp));
[iCpBetaOpt, iCpTSROpt] = find(T.Cp == T.CpMax);
T.TSROpt = T.CTSR(iCpTSROpt);
T.betaOpt = T.CPitch(iCpBetaOpt);

clear NDTSR NDPITCH Tables TSRMesh

%% Controller tunings
TSR_Kp = -0;
TSR_Ki = -50;
WSE_Kp = 10;
WSE_Ki = 1;

%% Undertainty in Cp-table
CpUncertaintyFactor = 1.0;

%% Simulate
open_system(simmdl)
sim(simmdl, TMax);

%% Plots
CqMax = max(T.Cq(:, iCpBetaOpt));
iCqMax = find(T.Cq(:, iCpBetaOpt) == CqMax);
CqMax_TSR = T.CTSR(iCqMax);
CqMax_ymax = ceil(CqMax*100)/100;

figure(1)
plot(T.CTSR, T.Cq(:,iCpBetaOpt)), hold on
plot([CqMax_TSR CqMax_TSR], [0 CqMax_ymax] , '--r');
xlabel('TSR [-]')
ylabel('C_q [-]')
ylim([0 CqMax_ymax])
title('Torque coefficient, stability boundary')
grid on

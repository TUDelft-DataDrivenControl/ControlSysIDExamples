clc, clear all, close all

deg2rad = pi/180;
Ts = 0.01;
Wr0 = 6*pi/30;
TMax = 1e5;
simmdl = 'WindTurbineESCKOpt';

% Load turbine constants
T = loadturbineconstants('NREL5MW');

% Load prior information on rotor performance (Cp, Ct, Cq)
load(['TablesCpCtCmCq.mat'])
T.Cp = Tables.Cp';
T.Ct = Tables.Ct';
T.CTSR = Tables.TSR;
T.CPitch = Tables.Pitch*deg2rad;

[NDTSR, NDPITCH] = ndgrid(Tables.TSR, Tables.Pitch);
T.CPfunc = griddedInterpolant(NDTSR, NDPITCH, Tables.Cp');
T.CTfunc = griddedInterpolant(NDTSR, NDPITCH, Tables.Ct');

% Find maximum Cp and corresponding TSR
T.CpMax = max(max(T.Cp));
[iCpBetaOpt, iCpTSROpt] = find(T.Cp == T.CpMax);
T.TSROpt = T.CTSR(iCpTSROpt);
T.betaOpt = T.CPitch(iCpBetaOpt);

clear NDTSR NDPITCH Tables iCpBetaOpt iCpTSROpt

T.K = optimalmodegain(T.rho, T.R, T.TSROpt, T.CpMax);

% ESC config
D_A = 3e5; 
D_w = 0.09;
M_phi = 0;
K_i = 1000;
wc_LPF = 0.1;
wc_HPF = 0.02;

% Simulate
open_system(simmdl)
sim(simmdl, TMax);
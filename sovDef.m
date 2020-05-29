clear; clc;
reset(gpuDevice);
rng(25);

%
% Parameters
%
R = 1.032;
beta = 0.9;
lambda0 = 0.6;
lambda1 = 0.64;
eta = 0.25;

ySz = 101;
bSz = 400;
bMin = 0.0;
bMax = 0.3;

%
% Endowment Process
%
[ y, Pi ] = makeMarkovChain(0.0, 0.9, 0.02, ySz);

% Grids
y = gpuArray(single(exp(y)));
gpuPi = gpuArray(single(Pi));

hy = y - max( 0.0, -lambda0 * y + lambda1 * y.^2);
uhy = ((1-beta) * (1.0 - 1.0 ./ hy))';

b = gpuArray(single(linspace(bMin, bMax, bSz)));

% Memory on GPU
Vd0 = gpuArray.zeros([ ySz, 1 ], 'single');
Vd1 = gpuArray.zeros([ ySz, 1 ], 'single');
Vr0 = gpuArray.zeros([ ySz, bSz ], 'single');
Vr1 = gpuArray.zeros([ ySz, bSz ], 'single');
V0 = gpuArray.zeros([ ySz, bSz ], 'single');
V1 = gpuArray.zeros([ ySz, bSz ], 'single');
q0 = gpuArray.zeros([ ySz, bSz ], 'single');
q1 = gpuArray.zeros([ ySz, bSz ], 'single');
bPrIx = gpuArray.ones([ ySz, bSz ], 'uint16');
dPol = gpuArray.ones([ ySz, bSz ], 'single');

%
% Iterate
%
errV = 1.0;
errQ = 1.0;
errTolV = 1.0e-8;
errTolQ = 1.0e-4;
iter = 1;
while errV > errTolV || errQ > errTolQ
    tic;
    
    %
    % Flow util. under repayment
    %
    [gpuB, gpuY, gpuA] = meshgrid(b', y', b');
    qq = repmat(q0, [1, 1, bSz ] );
    cc = gpuY + qq .* gpuB - gpuA;
    uu = (1-beta) * (1.0 - 1.0 / cc);
    uu(cc < 0.0) = -5000.0;
    
    %
    % Continuation value under repayment
    %
    contVal = repmat(gpuPi * V0, [1, 1, bSz ]);
    
    [ Vr1, bPrIx ] = max( uu + beta * contVal, [], 2 );
    Vr1 = squeeze(Vr1);
    bPrIx = squeeze(bPrIx);
    
    %
    % Value in default
    %
    Vd1 = uhy + beta * gpuPi * (eta * V0(:, 1) + (1-eta) * Vd0 );
    
    %
    % Default decision
    %
    tmp = repmat(Vd1, [1, bSz]);
    V1 = max( Vr1, tmp );
    dPol(:, :) = 0;
    dPol(Vr1 < tmp) = 1;
    
    %
    % Bond price schedule
    %
    q1 = gpuPi * (1-dPol) / R;
    
    errV = max(abs(V1(:)-V0(:)));
    V0 = V1;
    Vd0 = Vd1;

    errQ = max(abs(q1(:)-q0(:)));
    % q0 = 0.05 * q1 + 0.95 * q0;
    q0 = q1;
    
    elapsed = toc;
    if mod(iter, 5) == 0
        fprintf('%d ~ %8.6f & %8.6f ~ %8.6fs \n', ...
            iter, errV, errQ, elapsed);
    end
    iter = iter + 1;
end

% Done. Bring from GPU
b = double(gather(b));
V = double(gather(V1));
Vr = double(gather(Vr1));
Vd = double(gather(Vd1));
q = double(gather(q1));
bPrIx = gather(bPrIx);
dPol = gather(dPol);

%
% Policies and values
%

figure(1);
contourf(b, y, V); colorbar;
title('V');
xlabel('b');
ylabel('y');

figure(2);
contourf(b, y, q); colorbar;
title('q');
xlabel('b');
ylabel('y');

figure(3);
contourf(b, y, dPol, 2); colorbar;
title('d');
xlabel('b');
ylabel('y');

figure(4);
contourf(b, y, b(bPrIx), 15); colorbar;
title('b''');
xlabel('b');
ylabel('y');

figure(5);
plot(y, hy, 'b-');
hold on
plot(y, y, '--k');
title('h(y)');
xlabel('y');

%
% Simulation
%

simSz = 10000;
[ ySim, yIxSim ] = simulateMarkov( simSz, y, Pi );
bIxSim = ones([ 1, simSz ]);
dSim = zeros([ 1, simSz ]);
for t = 2:simSz
    if dSim(t-1) == 0
        dSim(t) = dPol(yIxSim(t), bIxSim(t-1));
    else
        if rand() < eta
            dSim(t) = 0;
        else
            dSim(t) = 1;
        end
    end
    
    if dSim(t) == 0
        bIxSim(t) = bPrIx(yIxSim(t), bIxSim(t-1));
    else
        bIxSim(t) = 1;
    end
end

figure(6);
yyaxis left; plot(1:300, ySim(1:300)); ylabel('y');
yyaxis right; plot(1:300, b(bIxSim(1:300))); ylabel('b''');
title('Simulation');
xlabel('Time');

figure(7);
histogram(b(bIxSim));
xlabel('b');

figure(8);
histogram(dSim);
xlabel('d');
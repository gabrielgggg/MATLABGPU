clear; clc;
reset(gpuDevice);

%
% Parameters
%
R = 1.03;
beta = 0.95;

bMin = 0.0;
bMax = 2.0;

ySz = 51;
bSz = 900;

%
% Endowment Process
%
[ y, Pi ] = makeMarkovChain(0.0, 0.95, 0.05, ySz);

% Grids
y = gpuArray(single(exp(y)));
b = gpuArray(single(linspace(bMin, bMax, bSz)));
gpuPi = gpuArray(single(Pi));

% Memory on GPU
V0 = gpuArray.zeros([ ySz, bSz ], 'single');
V1 = gpuArray.zeros([ ySz, bSz ], 'single');
bPrIx = gpuArray.ones([ ySz, bSz ], 'uint16');

% Pre-compute consumption & utils for all (y, b, b')
[gpuB, gpuY, gpuA] = meshgrid(b', y', b');
cc = gpuY - gpuB + R * gpuA;
uu = (1-beta) * (1.0 - 1.0 / cc);
uu(cc < 0.0) = -5000.0;

%
% Iterate
%
err = 1.0;
errTol = 1.0e-8;
iter = 1;
while err > errTol
    tic;
    contVal = repmat(gpuPi * V0, [1, 1, bSz ]);
    [ V1, bPrIx ] = max( uu + beta * contVal, [], 2 );
    
    V1 = squeeze(V1);
    bPrIx = squeeze(bPrIx);
    
    err = max(abs(V1(:)-V0(:)));
    V0 = V1;
    elapsed = toc;
    if mod(iter, 20) == 0
        fprintf('%d ~ %8.6f ~ %8.6fs \n', iter, err, elapsed);
    end
    iter = iter + 1;
end

% Done. Bring from GPU
b = double(gather(b));
V1 = double(gather(V1));
bPrIx = gather(bPrIx);

%
% Values and Policies
%

figure(1)
plot(b, b(bPrIx) - repmat(b, [ySz, 1]), '-'); hold on; plot(b, zeros(size(b)), '--k');
xlabel('Assets (b)');
ylabel('Change in Assets (b'' - b)');
title('Policy');

figure(2)
plot(b, V1, '-');
xlabel('Assets (b)');
title('Value Function');

%
% Stationary distribution with eigenvector of sparse matrix
%

% Collapse 2D to 1D
mapping = zeros([ ySz*bSz, 2 ]);
revMapping = zeros([ ySz bSz ]);
ix = 1;
for yIx = 1:ySz
    for bIx = 1:bSz
        mapping(ix, 1) = yIx;
        mapping(ix, 2) = bIx;
        revMapping(yIx, bIx) = ix;
        ix = ix + 1;
    end
end

ii = zeros([ySz * bSz * ySz, 1]);
jj = zeros([ySz * bSz * ySz, 1]);
vv = zeros([ySz * bSz * ySz, 1]);

ix = 1;
for bIx = 1:bSz
    for yIx = 1:ySz
        myFrom = revMapping(yIx, bIx);
        dest = bPrIx(yIx, bIx);
        for yPrIx = 1:ySz
            ii(ix) = myFrom;
            jj(ix) = revMapping(yPrIx, dest);
            vv(ix) = Pi(yIx, yPrIx);
            ix = ix + 1;
        end
    end
end

S = sparse(ii, jj, vv);
if size(S, 2) < size(S, 1)
  S(:, ySz * bSz) = 0.0;
end
% Eigenvector of Markov transition of (y, b) => (y', b')
[VV, DD] = eigs(S.', 1);

VV = VV ./ sum(VV);
stat = ones([ySz, bSz]);
for yIx = 1:ySz
    for bIx = 1:bSz
        stat(yIx, bIx) = VV(revMapping(yIx, bIx));
    end
end
% stat constains the stationary distribution

% figure(3);
% contourf(b(2:end), y, stat(:, 2:end), 15); colorbar;

fprintf('Agg savings %6.4f at r = %6.4f \n', dot( sum(stat, 1), b ), R-1);

function [ X, Pi ] = makeMarkovChain(yBar, rho, sgma, n)
%
% Construct Markov Chain with states X and transition probabilities Pi
% using AR(1) with mean yBar, autocorrelation rho, and s.d. of innovation
% sgma, with n points of support.
%

% X = equally spaced over an interval
% [ yBar - 2 s.d, yBar + 2 s.d. ]
X = linspace( yBar - 2 * sgma / sqrt(1 - rho^2), ...
    yBar + 2 * sgma / sqrt(1 - rho^2), n);

% Pi - transition matrix
Pi = zeros([ n, n ]);
for stYesterday = 1:n
  % work at state stYesterday
  % Left tail...
  Pi(stYesterday, 1) = normcdf((X(1) + X(2))/2, ...
      (1-rho) * yBar + rho * X(stYesterday), sgma);
  % Right tail...
  Pi(stYesterday, n) = 1 - normcdf((X(n-1) + X(n))/2, ...
      (1-rho) * yBar + rho * X(stYesterday), sgma);
  % Interior states...
  for stToday = 2:n-1
      Pi(stYesterday, stToday) = ...
          normcdf((X(stToday) + X(stToday+1))/2, ...
            (1-rho) * yBar + rho * X(stYesterday), sgma) ...
          - normcdf((X(stToday-1) + X(stToday))/2, ...
            (1-rho) * yBar + rho * X(stYesterday), sgma);
  end
end

end

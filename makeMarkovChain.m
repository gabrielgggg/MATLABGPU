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





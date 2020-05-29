function [ sample, sampleIx ] = simulateMarkov( T, X, Pi )
sampleIx = ones([ T, 1 ]);

% Get size of support of Markov Chain
n = size(Pi, 1);

% Initial/first value
sampleIx(1) = 1;

% All others
for tt = 2:T
    % Use conditional transition probabilities Pi(state yesterday, :)
    draw = rand();
    conditional = Pi( sampleIx(tt-1), : ); % PMF
    cumcond = cumsum(conditional); % CMF
    for stateToday = 1:n
        if draw < cumcond(stateToday) % found state with cummulative pr. greater than draw
            sampleIx(tt) = stateToday;
            break; % stop for, stop looking
        end
    end
end

% Get values associated with indices
sample = X(sampleIx);
end


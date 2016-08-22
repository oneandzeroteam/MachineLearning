function delta = crossEntrophyCost_delta(a, y)
    first_term = a - y; % gradient
    delta = first_term;
end
function delta = quadraticCost_delta(a, y)
    first_term = a - y; % gradient
    second_term = a .* (1-a); % sigmoid �̺а�, sigmoid = output_data_a
    delta = first_term .* second_term;
end
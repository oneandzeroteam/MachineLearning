function output = softmax(input)
    output = exp(input) / sum(exp(input));
end
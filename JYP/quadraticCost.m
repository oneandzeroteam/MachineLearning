function cost = quadraticCost(output_data_a, output_groundtruth_y)
    cost = mean((output_data_a - output_groundtruth_y).^2);
end
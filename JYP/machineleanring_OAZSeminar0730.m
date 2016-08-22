load('mnist_uint8_matlab.mat')

%% 하이퍼 파라미터 세팅
% 입력뉴런개수 / 출력뉴런개수 / 히든레이어개수 / 히든레이어뉴런개수 / 러닝레이트(7) / 반복회수
input_neuron_num = 784;
output_neuron_num = 10;
hidden_layer_num = 2;
hidden_neuron_num = [30, 20];
learningrate = 0.05;
epoch_num = 50; %iteration 횟수

%% 하이퍼 파라미터에따라 네트워크를 구성
weight_layer_num = hidden_layer_num + 1;
network_weight = cell(weight_layer_num, 1);
network_bias = cell(weight_layer_num, 1);

for layer_index = 1:weight_layer_num,
    if layer_index == 1,
        row_num = input_neuron_num;
        col_num = hidden_neuron_num(1);
    elseif layer_index == weight_layer_num
        row_num = hidden_neuron_num(hidden_layer_num);
        col_num = output_neuron_num;
    else
        row_num = hidden_neuron_num(layer_index -1);
        col_num = hidden_neuron_num(layer_index);
    end
    network_weight{layer_index, 1} = randn(row_num, col_num) / sqrt(input_neuron_num);
    network_bias{layer_index, 1} = randn(col_num, 1) / sqrt(input_neuron_num);
end

%% 알고리즘 중간변수
neuron_value = cell(weight_layer_num, 1);
delta_value = cell(weight_layer_num, 1);

for k1 = 1:epoch_num,
    %% BackPropagation algorithm
    % 1. input setting
    for input_index = 1:length(train_x)
        input_data = double(train_x(input_index, :)') / 255; % normalization (uint8 = 0 ~ 255)

        % 2. feed forward
        output_data_a = input_data;
        for layer_index = 1:weight_layer_num
            output_data_z = network_weight{layer_index, 1}' ...
                * output_data_a + network_bias{layer_index, 1};
            output_data_a = sigmoid(output_data_z);
            neuron_value{layer_index,1} = output_data_a;
        end

        % 3. output error
        output_ground_truth = double(train_y(input_index, :)'); % 10x1
        training_error = quadraticCost(output_data_a, output_ground_truth); % quadratic cost function
        % training_error = crossentropyCost(output_data_a, ...
        %                   output_ground_truth); % crossentropy cost f

        first_term = output_data_a - output_ground_truth; % gradient
        second_term = output_data_a .* (1-output_data_a); % sigmoid 미분값, sigmoid = output_data_a
        delta_value{weight_layer_num, 1} = first_term .* second_term;
        %delta value = error라고말하겠음

        % 4. Back Propagate error
        for layer_index = hidden_layer_num : -1 : 1
            first_term = network_weight{layer_index + 1, 1} ...
                * delta_value{layer_index + 1, 1};
            second_term = neuron_value{layer_index , 1} .* ...
                (1 - neuron_value{layer_index, 1});
            delta_value{layer_index, 1} = first_term .* second_term;
        end

        % 5. Output Gradient + Gradient Descent
        for layer_index = 1:weight_layer_num
            if layer_index == 1,
                weight_gradient = input_data * delta_value{layer_index, 1}';
            else
                weight_gradient = neuron_value{layer_index-1} * delta_value{layer_index, 1}';
            end
            bias_gradient = delta_value{layer_index, 1};

            network_weight{layer_index, 1} = network_weight{layer_index , 1} ...
                - learningrate * weight_gradient;
            network_bias{layer_index, 1} = network_bias{layer_index, 1} ...
                - learningrate * bias_gradient;
        end
    end
    
    %% Test set에 대한 성능측정
    predict_y = zeros(size(test_y));
    for input_index = 1:length(test_x)
        input_data = double(test_x(input_index, :)') / 255; % normalization (uint8 = 0 ~ 255)

        output_data_a = input_data;
        for layer_index = 1:weight_layer_num
            output_data_z = network_weight{layer_index, 1}' ...
                * output_data_a + network_bias{layer_index, 1};
            output_data_a = sigmoid(output_data_z);
        end
        predict_y(input_index, :) = (output_data_a == max(output_data_a))';
    end
    error_rate = sum(sum(abs(predict_y - double(test_y)))) / (2 * length(test_x));
    [k1, error_rate]
end





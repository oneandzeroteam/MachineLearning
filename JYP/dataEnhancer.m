function [train_mass_x, train_mass_y] = dataEnhancer(train_x, train_y)
%
iteration = 9;
%
train_mass_x = train_x;
train_mass_y = train_y;

for k1 = 1:iteration
    train_mass_x = dataMerge(train_mass_x, noiser(train_x));
    train_mass_y = dataMerge(train_mass_y, train_y);
end

end
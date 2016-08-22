function train_mass1_x = noiser(train_x)
[len1, len2] = size(train_x);
train_mass1_x = zeros(len1,len2);

for k1 = 1:len1
    pic = train_x(k1,:);
        
    %
    noise = ((pic == 0) .* (rand(size(pic)) < 0.05)) .* (rand(size(pic))*255);
    train_mass1_x(k1,:) = pic + uint8(noise);
    %
end

end
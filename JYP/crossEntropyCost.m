function cost = crossEntropyCost(a, y)
    cost = (y).*log(a) + (1-y).*log(1-a);
end
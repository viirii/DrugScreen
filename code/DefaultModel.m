classdef DefaultModel
        methods 
            function label = predict(~, x)
                label = ones(size(x, 1), 1) * -1;
            end
        end
end
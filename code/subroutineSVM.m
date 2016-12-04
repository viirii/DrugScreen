%% This function learns a SVM model from training data. It takes two inputs, S and T. The algorithm
% tries to find a model consisent with S. If it can't, it returns a flag.
% If it can, it returns a model that minimizes the error on T, while still
% being consistent with S.
% You do not need to change this
function [h,flag] = subroutineSVM(S, T, Slabels, Tlabels)
% Input:  S - a 1 by n vector containing the training instances in S
%         T - a 1 by m vector containing the training instances in T
%         SLabels - a 1 by n vector containing the training labels for S
%         TLabels - a 1 by m vector containing the training labels for T
% Output: h - a scalar containinng the model [threshold]
%         flag - a scalar that is 0 if the algorithm succeeds, 1 otherwise
flag = 1;
h = DefaultModel(); % default model, everything is zero

if(length(S)==0 && length(T)==0) % S and T are empty, return default model
    flag = 0;
    return
elseif(length(S)==0) % S is empty, T is not; model is determined by T
    flag = 0;
    if(length(find(Tlabels==1))==0) % there are no positive examples in T, return default model
        % do nothing
    else % find a model that fits T and minimizing the error
        h = fitclinear(T, Tlabels, 'Learner', 'svm');
    end
    return
elseif(length(T)==0) % T is empty, S is not; model is determined by S
    if(length(find(Slabels==1))==0) % there are no positive examples in S; use default model
        flag = 0;
    else % confirm model is consistent
        h = fitclinear(S, Slabels, 'Learner', 'svm');
        if(sum(h.predict(S) - Slabels) == 0) % only keep models that are consistent
            flag = 0;
        end
    end
    return
else % S and T have elements in them
    if(length(find(Slabels==1))==0 && length(find(Tlabels==1))==0) % neither set has positive examples, use default model
        flag = 0;
%     elseif(length(find(Slabels==1))==0) % S doesnt have positive examples, T does
%         flag = 0;
%         
%         
%     elseif(length(find(Tlabels==1))==0) % T doesnt have positive examples, S does
%         h = min(S(find(Slabels==1))); % left-most point in S labeled 1
%         if(isconsistent(S,Slabels,h)==1) % only keep models that are consistent
%             flag = 0;
%         end
    else % both have positive elements
    modifier = 1;
%     while (true)
        weights = [ones(size(Slabels, 1), 1) * 1; ones(size(Tlabels, 1), 1)];
        h = fitclinear([S; T], [Slabels; Tlabels], 'Learner', 'svm', 'Weights', weights);
%         if(sum(h.predict(S) - Slabels) == 0) % only keep models that are consistent
            flag = 0;
%             return
%         end
%         modifier = modifier + 1;
%         sprintf('modifier increased to %d', modifier)
%     end
        
    end
    return
end
end

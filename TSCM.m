function [pvalue,CMN,CMNstar] = TSCM(JhatN,JNstar,N,B)

if iscell(JNstar) == 1
    CM_n = cellfun(@(x)diag(x*x'), JNstar,'UniformOutput',false);
    CM_n = cell2mat(CM_n); %B x N
    CMNstar = mean(CM_n,2);
else
    CMNstar = mean(JNstar.^2,2);
end

CMN = trace(JhatN*JhatN')/N;
    
pvalue = sum(CMNstar>=CMN)/B;
    
end
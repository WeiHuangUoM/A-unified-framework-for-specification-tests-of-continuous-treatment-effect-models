function [pvalue,KSN,KSNstar] = TSKS(JhatN,JNstar,B)

if iscell(JNstar) == 1
    KS_n = cellfun(@(x)max(abs(x),[],2), JNstar,'UniformOutput',false);
    KS_n = cell2mat(KS_n); %B x Nt
    KSNstar = max(KS_n,[],2);
else
    KSNstar = max(abs(JNstar),[],2);
end
KSN = max(abs(JhatN),[],'all');
    
pvalue = sum(KSNstar>=KSN)/B;
    
end
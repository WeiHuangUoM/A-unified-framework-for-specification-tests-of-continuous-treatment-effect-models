function H = Hfunction(Ti,t,Htype)
 Ti = reshape(Ti,length(Ti),1);
 t = reshape(t,1,length(t));

 if strcmp(Htype,'exp')==1
 H = exp(Ti*t);
 elseif strcmp(Htype, 'Logit')==1
 H = 1./(1+exp(5-Ti*t));
 elseif strcmp(Htype,'Sine') ==1
 H = cos(Ti*t)+sin(Ti*t);
 elseif strcmp(Htype,'Indicator')==1
 H = Ti<=t;
 end
end
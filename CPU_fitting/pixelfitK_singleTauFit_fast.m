function fittingresults = pixelfitK_singleTauFit_fast(Texp, K, functiontype)
    switch functiontype
        case 't'
            fun = fittype(('beta^0.5*(rho^2*(exp(-2*(Texp/tauC))-1+2*(Texp/tauC))/(2*(Texp/tauC)^2)+4*rho*(1-rho)*(exp(-(Texp/tauC))-1+(Texp/tauC))/((Texp/tauC)^2))^0.5'),'indep','Texp');
        case 's'
            fun = fittype(('beta^0.5*(rho^2*(exp(-2*(Texp/tauC))-1+2*(Texp/tauC))/(2*(Texp/tauC)^2)+4*rho*(1-rho)*(exp(-(Texp/tauC))-1+(Texp/tauC))/((Texp/tauC)^2)+(1-rho)^2)^0.5'),'indep','Texp');
    end
    lb = [0.72 0 10e-3]; % [beta rho tauC]
    ub = [0.72 1 20000]; % ms
    Fopts = fitoptions(fun);
    Fopts.Lower = lb;
    Fopts.Upper = ub;
    Fopts.Display='off';
    Fopts.TolFun=1e-6;
    Fopts.Robust='LAR';
    Fopts.StartPoint=[0.5 0.5 1];% [beta rho tauC]
    est = fit(Texp',K,fun,Fopts);
    fittingresults.varFit = [est.beta, est.rho, est.tauC];
    fittingresults.KFit = fun(est.beta, est.rho, est.tauC, Texp');
    fittingresults.R = (1-sum(abs(K-fittingresults.KFit).^2)./sum(abs(K-mean(K)).^2));
end
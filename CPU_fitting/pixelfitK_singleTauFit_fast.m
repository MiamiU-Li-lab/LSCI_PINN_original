function fittingresults = pixelfitK_singleTauFit_fast(Texp, K, functiontype)
    switch functiontype
        case 't'
            fun = fittype(('beta^0.5*(rho^2*(exp(-2*sqrt(Texp/tauC)).*(4*(Texp/tauC) + 6*sqrt(Texp/tauC) + 3)-3+2*(Texp/tauC))/(2*(Texp/tauC)^2)+8*rho*(1-rho)*(exp(-sqrt(Texp/tauC)).*(2*(Texp/tauC) + 6*sqrt(Texp/tauC) + 6)-6+(Texp/tauC))/((Texp/tauC)^2))^0.5'),'indep','Texp');
        case 's'
            fun = fittype(('beta^0.5*(rho^2*(exp(-2*sqrt(Texp/tauC)).*(4*(Texp/tauC) + 6*sqrt(Texp/tauC) + 3)-3+2*(Texp/tauC))/(2*(Texp/tauC)^2)+8*rho*(1-rho)*(exp(-sqrt(Texp/tauC)).*(2*(Texp/tauC) + 6*sqrt(Texp/tauC) + 6)-6+(Texp/tauC))/((Texp/tauC)^2)+(1-rho)^2)^0.5'),'indep','Texp');
    end
    lb = [0.72 0 10e-3]; % [beta rho tauC] % normal incidence LSCI 0.82^2; oblique incidence LSCI: 0.709^2
    ub = [0.72 1 20000]; % ms % Updated normal incidence LSCI 0.847^2 before 0709 2024; updated 0709 2024 0.85; updated 0719 2024 0.8362
    Fopts = fitoptions(fun);
    Fopts.Lower = lb;
    Fopts.Upper = ub;
    Fopts.Display='off';
    Fopts.TolFun=1e-6;
    Fopts.Robust='LAR';
    Fopts.StartPoint=[0.72 0.5 1];% [beta rho tauC]
    est = fit(Texp',K,fun,Fopts);
    fittingresults.varFit = [est.beta, est.rho, est.tauC];
    fittingresults.KFit = fun(est.beta, est.rho, est.tauC, Texp');
    fittingresults.R = (1-sum(abs(K-fittingresults.KFit).^2)./sum(abs(K-mean(K)).^2));
end

function [varFit, R] = pixelfitK_GPU_Vectorized_fast(Texp, K_matrix, functiontype)
% PIXELFITK_SINGLETAUFIT_FAST Vectorized Trust-Region-Reflective optimizer on GPU.
% Uses IRLS with LAR weights (w = 1/|r|) to reproduce MATLAB fit(...,'Robust','LAR').
% Tracks all matrix pixels simultaneously using exact analytical derivatives.

    [nT, num_pixels] = size(K_matrix);

    % 1. Initialization Parameters Per Pixel
    beta  = ones(1, num_pixels, 'gpuArray') * 0.72;  % fixed beta, matches PINN and CPU fast fitter
    rho   = ones(1, num_pixels, 'gpuArray') * 0.5;
    tauC  = ones(1, num_pixels, 'gpuArray') * 1.0;

    % Trust-Region Hyperparameters
    max_iters = 400;
    Delta     = ones(1, 1, num_pixels, 'gpuArray') * 1.0;  % initial trust radius [1x1xP]
    Delta_max = 1e4;
    eta       = 0.1;   % minimum ratio threshold for step acceptance
    TolFun    = 1e-6;  % termination tolerance on cost (matches MATLAB fit default)

    % Format inputs to uniform 3D layouts for clean pagefun operations
    T_3D = reshape(Texp, nT, 1, 1);
    K_3D = reshape(K_matrix, nT, 1, num_pixels);

    % Initial evaluation before loop
    [K_pred, J_rho, J_tauC] = evaluate_fast_jacobian(T_3D, beta, rho, tauC, functiontype);
    residuals = K_3D - K_pred;
    old_cost  = sum(abs(residuals), 1);   % LAR cost: sum|r|, [1x1xP]

    % 2. Optimization Loop (Runs globally over all pixels simultaneously)
    for iter = 1:max_iters
        % LAR (IRLS) weights: w_i = 1 / max(|r_i|, eps) [nT x 1 x num_pixels]
        weights = 1 ./ max(abs(residuals), 1e-6);

        % Assemble 3D Jacobian and apply IRLS weights: [nT x 2 x num_pixels]
        J   = cat(2, J_rho, J_tauC);
        J_W = J .* weights;

        % Weighted normal equations (J^T*W*J and J^T*W*r)
        JTJ = pagefun(@mtimes, pagetranspose(J_W), J);           % [2x2xP]
        JTr = pagefun(@mtimes, pagetranspose(J_W), residuals);   % [2x1xP]

        % Gauss-Newton step — no LM damping; trust radius constrains the step
        step = pagefun(@mtimes, pageinv(JTJ), JTr);   % [2x1xP]

        % Trust-region constraint: scale step to lie within Delta ball
        step_norm = sqrt(sum(step.^2, 1));                       % [1x1xP]
        scale     = min(1.0, Delta ./ max(step_norm, 1e-12));    % [1x1xP]
        step      = step .* scale;                                % [2x1xP]

        % Trial parameters
        rho_t  = rho  + reshape(step(1,1,:), 1, num_pixels);
        tauC_t = tauC + reshape(step(2,1,:), 1, num_pixels);

        % Reflective bounds (physically constrained, matches MATLAB TRR)
        rho_t  = reflect_bounds(rho_t,  0.0,   1.0);
        tauC_t = reflect_bounds(tauC_t, 1e-2,  20000.0);

        % Evaluate model at trial point
        [K_t, J_rho_t, J_tauC_t] = evaluate_fast_jacobian(T_3D, beta, rho_t, tauC_t, functiontype);
        res_t    = K_3D - K_t;
        new_cost = sum(abs(res_t), 1);   % [1x1xP]

        % Trust-region ratio: actual / predicted reduction in LAR cost
        % predicted = step^T * JTr - 0.5 * step^T * JTJ * step  (quadratic model)
        pred_num  = pagefun(@mtimes, pagetranspose(step), JTr);                          % [1x1xP]
        pred_quad = pagefun(@mtimes, pagetranspose(step), pagefun(@mtimes, JTJ, step));  % [1x1xP]
        predicted = pred_num - 0.5 .* pred_quad;                                         % [1x1xP]
        actual    = old_cost - new_cost;                                                  % [1x1xP]
        ratio     = actual ./ max(abs(predicted), 1e-12);                                % [1x1xP]

        % TolFun: stop when no pixel improved by more than TolFun in this step
        if gather(max(abs(actual(:)))) < TolFun
            break;
        end

        % Accept/reject step per pixel (branchless, ratio > eta accepts)
        accept = (ratio > eta);                         % [1x1xP]
        a2d    = reshape(accept, 1, num_pixels);        % [1xP]  for 2D params
        rho    = a2d .* rho_t  + (~a2d) .* rho;
        tauC   = a2d .* tauC_t + (~a2d) .* tauC;

        % Blend K_pred, Jacobians, residuals and cost at accepted point
        K_pred    = accept .* K_t      + (~accept) .* K_pred;
        J_rho     = accept .* J_rho_t  + (~accept) .* J_rho;
        J_tauC    = accept .* J_tauC_t + (~accept) .* J_tauC;
        residuals = K_3D - K_pred;
        old_cost  = accept .* new_cost + (~accept) .* old_cost;

        % Update trust radius based on model quality ratio (branchless)
        shrink = (ratio < 0.25);
        expand = (ratio > 0.75) & accept;
        Delta  = (shrink  .* (0.25 .* Delta)) + ...
                 (expand  .* min(2.0 .* Delta, Delta_max)) + ...
                 (~shrink & ~expand) .* Delta;
    end

    % 3. Format Output Matrix [3 x num_pixels]
    varFit = [beta; rho; tauC];

    % Compute global R-squared map array
    K_mean = sum(K_matrix, 1) / nT;
    tot_ss = sum((K_matrix - K_mean).^2, 1);
    res_ss = reshape(sum((K_3D - K_pred).^2, 1), 1, num_pixels);
    R = 1 - (res_ss ./ tot_ss);
end


%% --- REFLECTIVE BOUNDS HELPER ---

function x = reflect_bounds(x, lb, ub)
% Reflects x back into [lb, ub] using periodic folding (matches MATLAB TRR bounds).
    range = ub - lb;
    x     = mod(x - lb, 2 .* range);
    x     = lb + range - abs(x - range);
end


%% --- FAST ANALYTICAL JACOBIAN CORE HELPER ---

function [K, J_rho, J_tauC] = evaluate_fast_jacobian(T, beta, rho, tauC, type)
    rho_3D  = reshape(rho, 1, 1, []);
    tau_3D  = reshape(tauC, 1, 1, []);
    beta_3D = reshape(beta, 1, 1, []);
    
    x = T ./ tau_3D;
    sqrt_x = sqrt(x);
    exp_neg_sqrt_x  = exp(-sqrt_x);
    exp_neg_2sqrt_x = exp(-2 .* sqrt_x);
    
    % Model subfunctions matching pixelfitK_singleTauFit_fast.m
    f1 = (exp_neg_2sqrt_x .* (4 .* x + 6 .* sqrt_x + 3) - 3 + 2 .* x) ./ (2 .* x.^2);
    f2 = (exp_neg_sqrt_x  .* (2 .* x + 6 .* sqrt_x + 6) - 6 + x) ./ (x.^2);
    
    % Derivatives of subfunctions with respect to internal ratio x
    dN1_dx = 2 - 2 .* exp_neg_2sqrt_x .* (1 + 2 .* sqrt_x);
    df1_dx = (dN1_dx ./ (2 .* x.^2)) - ...
             ((exp_neg_2sqrt_x .* (4 .* x + 6 .* sqrt_x + 3) - 3 + 2 .* x) ./ (x.^3));

    dN2_dx = 1 - exp_neg_sqrt_x .* (1 + sqrt_x);
    df2_dx = (dN2_dx ./ (x.^2)) - ...
             (2 .* (exp_neg_sqrt_x .* (2 .* x + 6 .* sqrt_x + 6) - 6 + x) ./ (x.^3));
    
    % Compute variance based on target spatial or temporal formulation
    if strcmp(type, 't')
        V = (rho_3D.^2 .* f1) + (8 .* rho_3D .* (1 - rho_3D) .* f2);
        dV_drho = (2 .* rho_3D .* f1) + (8 .* (1 - 2.*rho_3D) .* f2);
    else
        f3 = (1 - rho_3D).^2;
        V = (rho_3D.^2 .* f1) + (8 .* rho_3D .* (1 - rho_3D) .* f2) + f3;
        dV_drho = (2 .* rho_3D .* f1) + (8 .* (1 - 2.*rho_3D) .* f2) - 2.*(1 - rho_3D);
    end
    
    % Outer function evaluate
    K = sqrt(beta_3D) .* sqrt(V);
    
    % Safeguard zero points to completely protect against division by zero errors
    K_safeguard = max(K, 1e-8);
    dK_dV = 0.5 .* beta_3D ./ K_safeguard;
    
    % Final analytical Jacobians via exact chain rule expansions
    J_rho  = dK_dV .* dV_drho;
    dV_dx  = (rho_3D.^2 .* df1_dx) + (8 .* rho_3D .* (1 - rho_3D) .* df2_dx);
    dx_dtauC = -T ./ (tau_3D.^2);
    J_tauC = dK_dV .* dV_dx .* dx_dtauC;
end

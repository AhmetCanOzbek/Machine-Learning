function[analytic_component] = get_analytic_component(peak_info,hrtbt_no)

    P_i = peak_info{1};
    P_amp = peak_info{2};
    R_i = peak_info{3};
    R_amp = peak_info{4};
    S_i = peak_info{5};
    S_amp = peak_info{6};
    T_i = peak_info{7};
    T_amp = peak_info{8};
    Q_i = peak_info{9};
    Q_amp = peak_info{10};
    
    %disp('Pi:'); disp(P_i);
    
    i = hrtbt_no;
    %Temporal
    Pi_Qi = abs(P_i(i) - Q_i(i));
    Pi_Ri = abs(P_i(i) - R_i(i));
    Pi_Si = abs(P_i(i) - S_i(i));
    Pi_Ti = abs(P_i(i) - T_i(i));        
    
    Qi_Ri = abs(Q_i(i) - R_i(i));
    Qi_Si = abs(Q_i(i) - S_i(i));
    Qi_Ti = abs(Q_i(i) - T_i(i));
    
    Ri_Si = abs(R_i(i) - S_i(i));    
    Ri_Ti = abs(R_i(i) - T_i(i));
    
    Si_Ti = abs(S_i(i) - T_i(i));
    
    
    %Amplitude
    P_Q = P_amp(i) - Q_amp(i);
    P_R = P_amp(i) - R_amp(i);
    P_S = P_amp(i) - S_amp(i);
    P_T = P_amp(i) - T_amp(i);
    
    Q_R = Q_amp(i) - R_amp(i);
    Q_S = Q_amp(i) - S_amp(i);
    Q_T = Q_amp(i) - T_amp(i);       
    
    R_S = R_amp(i) - S_amp(i);
    R_T = R_amp(i) - T_amp(i);   
    
    S_T = S_amp(i) - T_amp(i);
    
    
    
    temporal = [Pi_Qi, Pi_Ri, Pi_Si, Pi_Ti, Qi_Ri, Qi_Si, Qi_Ti, Ri_Si, Ri_Ti, Si_Ti];
    amplitude = [P_Q, P_R, P_S, P_T, Q_R, Q_S, Q_T, R_S, R_T, S_T];

    
    analytic_component = [temporal, amplitude];

end
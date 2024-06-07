using DifferentialEquations

# functions
evap_func(soilwater, en, x1) = soilwater * (2.0 - soilwater / x1) * tanh(en / x1) / (1.0 + (1.0 - soilwater / x1) * tanh(en / x1))
ps_func(soilwater, pn, x1) = x1 * (1.0 - (soilwater / x1)^2.0) * tanh(pn / x1) / (1.0 + soilwater / x1 * tanh(pn / x1))
perc_func(soilwater, x1, β) = soilwater * (1.0 - (1.0 + (soilwater / (β * x1))^4)^(-1 / 4))
exch_func(rts, x2, x3) = x2 * ((rts / x3)^3.5)
qr_func(rts, x3) = rts * (1.0 - (1.0 + (rts / x3)^4)^(-1 / 4))

function base_gr4h_run(input, pas, timeidx)
    x1, x2, x3, x4, β = pas[:params]
    soilwater_0, rtgstore_0 = pas[:u0][:soilwater] * x1, pas[:u0][:rtgstore] * x3
    prcp_vec, pet_vec = input
    n_uh = 480

    #* 1.build first ode equation
    function gr4h_soil_func(u, p, t)
        x1, β = p
        soilwater = u[1]
        prcp, pet = prcp_vec[Int(t)], pet_vec[Int(t)]
        en = pet - min(prcp, pet)
        pn = prcp - min(prcp, pet)
        evap = evap_func(soilwater, en, x1)
        ps = ps_func(soilwater, pn, x1)
        perc = perc_func(soilwater, x1, β)

        return [u[1] + ps - evap - perc]
    end

    #* 2.solve the first ode equation
    prob_1 = DiscreteProblem(gr4h_soil_func, [soilwater_0], (timeidx[1], timeidx[end]), ())
    sol_1 = solve(prob_1, FunctionMap(), u0=[soilwater_0], p=[x1, β])

    if !SciMLBase.successful_retcode(sol_1)
        println("sol 1 wrong")
        return ones(length(prcp_vec)) .* 100, ()
    end

    #* 3.get the inner variables
    soilwater_vec = sol_1[1, :]
    en_vec = pet_vec .- min.(prcp_vec, pet_vec)
    pn_vec = prcp_vec .- min.(prcp_vec, pet_vec)
    evap_vec = evap_func.(soilwater_vec, en_vec, x1)
    perc_vec = perc_func.(soilwater_vec, x1, β)
    ps_vec = ps_func.(soilwater_vec, pn_vec, x1)
    pr_vec = pn_vec .- ps_vec .+ perc_vec

    slowflow_vec = pr_vec .* 0.9
    fastflow_vec = pr_vec .* 0.1

    #* 5.get the routed slowflow and fastflow
    slowflow_vec = flux_unit_routing(slowflow_vec, timeidx, uh_1_half, x4, n_uh)
    fastflow_vec = flux_unit_routing(fastflow_vec, timeidx, uh_2_full, 2 * x4, 2 * n_uh)

    #* 6.build second ode equation
    function gr4h_frw_func(u, p, t)
        x2, x3 = p
        rts = u[1]
        slowflow = slowflow_vec[Int(t)]
        # slowflow = slow_itp(t)
        exch = exch_func(rts, x2, x3)
        qr = qr_func(max(rts + slowflow + exch, 0.0), x3)
        return [max(0.0, rts + slowflow + exch - qr)]
    end

    prob_2 = DiscreteProblem(gr4h_frw_func, [rtgstore_0], (timeidx[1], timeidx[end]), ())

    sol_2 = solve(prob_2, FunctionMap(), u0=[rtgstore_0], p=[x2, x3],
        sensealg=BacksolveAdjoint(autojacvec=ZygoteVJP())
    )

    if !SciMLBase.successful_retcode(sol_2)
        println("sol 2 wrong")
        return ones(length(prcp_vec)) .* 100, ()
    end

    rts_vec = sol_2[1, :]
    exch_vec = exch_func.(rts_vec, x2, x3)
    qr_vec = qr_func.(rts_vec .+ slowflow_vec .+ exch_vec, x3)
    qd_vec = max.(0.0, fastflow_vec .+ exch_vec)
    q_vec = max.(0.0, qd_vec .+ qr_vec)

    fluxes = (prcp=prcp_vec, pet=pet_vec, satu=ps_vec, evap=evap_vec, pr=pr_vec,
        perc=perc_vec, slowflow=slowflow_vec, flow=q_vec, qd=qd_vec,
        fastflow=fastflow_vec, routeflow=qr_vec, recharge=exch_vec,
        soilwater=sol_1[1, :], rtgstore=sol_2[1, :])

    q_vec, fluxes
end

export base_gr4h_run
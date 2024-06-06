import torch
from torch import tanh
from torch import tensor


def tensor32(value):
    return tensor(value, requires_grad=True).to(torch.float32)


def uh_h(x4, ss_fun, nh=480):
    uh = torch.zeros(nh)
    for i in range(1, nh):
        uh[i] = ss_fun(i, x4) - ss_fun(i - 1, x4)
    return uh


def ss1_h(i, x4, d=tensor32(1.25)):
    if i <= tensor32(0.0):
        return tensor32(0.0)
    elif i < x4:
        return (i / x4) ** d
    else:
        return tensor32(1.0)


def ss2_h(i, x4, d=tensor32(1.25)):
    if i <= tensor32(0.0):
        return tensor32(0.0)
    elif i <= x4:
        return tensor32(0.5) * (i / x4) ** d
    elif i < tensor32(2.0) * x4:
        return tensor32(1.0) - tensor32(0.5) * (tensor32(2.0) - i / x4) ** d
    else:
        return tensor32(1.0)


def route_func(q, uh_w, uh_u):
    q_route = []
    for i in range(len(q)):
        uh_u = uh_u.roll(-1, 0)
        uh_u[-1] = tensor32(0.0)
        uh_u = q[i] * uh_w + uh_u
        q_route.append(uh_u[0].view(1))
    q_route_arr = torch.cat(q_route, dim=0)
    q_route_arr = q_route_arr.roll(-1, 0)
    q_route_arr[-1] = tensor32(0.0)
    return q_route_arr, uh_u


def gr4h_slw_step(slw, prcp, pet, params):
    # Define the parameters of the model
    x1, x2, x3, x4, beta = params
    en = pet - torch.min(prcp, pet)
    pn = prcp - torch.min(prcp, pet)

    evap = slw * (tensor32(2.0) - slw / x1) * tanh(en / x1) / (
            tensor32(1.0) + (tensor32(1.0) - slw / x1) * tanh(en / x1))
    ps = x1 * (tensor32(1.0) - (slw / x1) ** tensor32(2.0)) * tanh(pn / x1) / (tensor32(1.0) + slw / x1 * tanh(pn / x1))
    perc = slw * (tensor32(1.0) - (tensor32(1.0) + (slw / (beta * x1)) ** tensor32(4.0)) ** (
            tensor32(-1.0) / tensor32(4.0)))

    # routed water
    pr = pn - ps + perc

    new_slw = slw + ps - evap - perc
    return new_slw, [pn, evap, ps, perc, pr]


def gr4h_rgt_step(rts, q9, q1, params):
    x1, x2, x3, x4, beta = params
    exch = x2 * ((rts / x3) ** tensor32(3.5))

    if rts + q9 + exch > tensor32(0.0):
        fl = -rts - q9
    else:
        fl = exch

    if q1 + exch > tensor32(0.0):
        fg = -q1
    else:
        fg = exch

    new_rts = torch.max(tensor32(0.0), rts + q9 + exch)
    qr = new_rts * (tensor32(1.0) - (tensor32(1.0) + (new_rts / x3) ** tensor32(4.0)) ** tensor32(-1 / 4))

    new_rts = new_rts - qr
    qd = torch.max(tensor32(0.0), q1 + exch)
    q = torch.max(tensor32(0.0), qr + qd)
    return new_rts, [q, qr, qd, fl, fg, exch]


def gr4h_core(input, params, initstates):
    prcp, pet = input
    x1, x2, x3, x4, beta = params

    nh = 480
    slw, rts = initstates
    time_len = len(prcp)
    uh_q9_w, uh_q1_w = uh_h(x4, ss1_h, nh=nh), uh_h(2 * x4, ss1_h, nh=2 * nh)
    uh_q9_u, uh_q1_u = torch.zeros(nh), torch.zeros(2 * nh)

    pr_list = []
    pn_list = []
    evap_list = []
    ps_list = []
    perc_list = []
    slw_list = []
    for i in range(time_len):
        slw, [pn, evap, ps, perc, pr] = gr4h_slw_step(slw, prcp[i], pet[i], (x1, x2, x3, x4, beta))
        slw_list.append(slw.view(1))
        pr_list.append(pr.view(1))
        pn_list.append(pn.view(1))
        evap_list.append(evap.view(1))
        ps_list.append(ps.view(1))
        perc_list.append(perc.view(1))

    pr_arr = torch.cat(pr_list, dim=0)
    pn_arr = torch.cat(pn_list, dim=0)
    slw_arr = torch.cat(slw_list, dim=0)
    evap_arr = torch.cat(evap_list, dim=0)
    ps_arr = torch.cat(ps_list, dim=0)
    perc_arr = torch.cat(perc_list, dim=0)
    q9_arr = pr_arr * tensor32(0.9)
    q1_arr = pr_arr * tensor32(0.1)

    q9_route_arr, uh_q9_u = route_func(q9_arr, uh_q9_w, uh_q9_u)
    q1_route_arr, uh_q1_u = route_func(q1_arr, uh_q1_w, uh_q1_u)

    q_list = []
    qr_list = []
    qd_list = []
    fl_list = []
    fg_list = []
    exch_list = []
    rts_list = []
    for i in range(time_len):
        rts, [q, qr, qd, fl, fg, exch] = gr4h_rgt_step(rts, q9_route_arr[i], q1_route_arr[i],
                                                       (x1, x2, x3, x4, beta))
        q_list.append(q.view(1))
        qr_list.append(qr.view(1))
        qd_list.append(qd.view(1))
        fl_list.append(fl.view(1))
        fg_list.append(fg.view(1))
        exch_list.append(exch.view(1))
        rts_list.append(rts.view(1))
    q_arr = torch.cat(q_list, dim=0)
    qr_arr = torch.cat(qr_list, dim=0)
    qd_arr = torch.cat(qd_list, dim=0)
    fl_arr = torch.cat(fl_list, dim=0)
    fg_arr = torch.cat(fg_list, dim=0)
    exch_arr = torch.cat(exch_list, dim=0)
    rts_arr = torch.cat(rts_list, dim=0)
    return (q_arr,
            [slw_arr, rts_arr],
            [pn_arr, evap_arr, ps_arr, perc_arr, pr_arr],
            [q9_route_arr, q1_route_arr],
            [qr_arr, qd_arr, fl_arr, fg_arr, exch_arr])

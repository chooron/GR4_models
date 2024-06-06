import numpy as np
import pandas as pd
import plotly.graph_objs as go
import torch

from gr4h import gr4h_core, tensor32

df = pd.read_pickle(r'../data/L0123003.pkl')
r_run_df = pd.read_csv(r'../data/airGR_sim.csv')
prcp_arr = torch.from_numpy(df['P'].values.astype(np.float32))
pet_arr = torch.from_numpy(df['E'].values.astype(np.float32))

(q_arr, [slw_arr, rts_arr],
 [pn_arr, evap_arr, ps_arr, perc_arr, pr_arr],
 [q9_route_arr, q1_route_arr],
 [qr_arr, qd_arr, fl_arr, fg_arr, exch_arr]) = gr4h_core(
    [prcp_arr, pet_arr],
    [tensor32(521.1), tensor32(-2.9), tensor32(218.0), tensor32(4.12), tensor32(21 / 4)],
    [tensor32(0.3 * 521.1), tensor32(0.5 * 218.0)],
)
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(len(df))), y=q1_route_arr.detach().numpy(), mode='lines', name='q_hat'))
fig.add_trace(go.Scatter(x=list(range(len(df))), y=r_run_df['q1'], mode='lines', name='q_obs'))
fig.write_html('prediction.html')

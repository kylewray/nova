model;


set STATES;
set ACTIONS;
set OBSERVATIONS;
set CONTROLLER_NODES;


param q0 {q in CONTROLLER_NODES} default 0.0, >= 0.0, <= 1.0;
param b0 {s in STATES} default 0.0, >= 0.0, <= 1.0;
param T {s in STATES, a in ACTIONS, sp in STATES} default 0.0, >= 0.0, <= 1.0;
param O {a in ACTIONS, s in STATES, o in OBSERVATIONS} default 0.0, >= 0.0, <= 1.0;
param R {s in STATES, a in ACTIONS} default 0.0;
param gamma default 0.95, >= 0.0, <= 1.0;


var V {CONTROLLER_NODES, STATES};
var psi {CONTROLLER_NODES, ACTIONS} >= 0.0, <= 1.0;
var eta {CONTROLLER_NODES, ACTIONS, OBSERVATIONS, CONTROLLER_NODES} >= 0.0, <= 1.0;


maximize Value:
    sum {q in CONTROLLER_NODES, s in STATES} q0[q] * b0[s] * V[q, s];


subject to Bellman_Constraint_V {q in CONTROLLER_NODES, s in STATES}:
    V[q, s] = sum {a in ACTIONS} (psi[q, a] * (R[s, a] + gamma * sum {sp in STATES} (T[s, a, sp] * sum {o in OBSERVATIONS} (O[a, sp, o] * sum {qp in CONTROLLER_NODES} (eta[q, a, o, qp] * V[qp, sp])))));


subject to Probability_Constraint_Psi_Nonnegative {q in CONTROLLER_NODES, a in ACTIONS}:
    psi[q, a] >= 0.0;

subject to Probability_Constraint_Psi_Normalization {q in CONTROLLER_NODES}:
    sum {a in ACTIONS} psi[q, a] = 1.0;

subject to Probability_Constraint_Eta_Nonnegative {q in CONTROLLER_NODES, a in ACTIONS, o in OBSERVATIONS, qp in CONTROLLER_NODES}:
    eta[q, a, o, qp] >= 0.0;

subject to Probability_Constraint_Eta_Normalization {q in CONTROLLER_NODES, a in ACTIONS, o in OBSERVATIONS}:
    sum {qp in CONTROLLER_NODES} eta[q, a, o, qp] = 1.0;


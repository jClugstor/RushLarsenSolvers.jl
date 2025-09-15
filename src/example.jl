using RushLarsenSolvers: RushLarsen, RushLarsenFunction
using SimpleDiffEq


function alpha_n(V)
    return 0.01 * (V + 55.0) / (1.0 - exp(-(V + 55.0) / 10.0))
end

function beta_n(V)
    return 0.125 * exp(-(V + 65.0) / 80.0)
end

function n_rates(V)
    return (alpha_n(V), beta_n(V))
end

function alpha_m(V)
    return 0.1 * (V + 40.0) / (1.0 - exp(-(V + 40.0) / 10.0))
end

function beta_m(V)
    return 4.0 * exp(-(V + 65.0) / 18.0)
end

function m_rates(V)
    return (alpha_m(V), beta_m(V))
end

function alpha_h(V)
    return 0.07 * exp(-(V + 65.0) / 20.0)
end

function beta_h(V)
    return 1.0 / (1.0 + exp(-(V + 35.0) / 10.0))
end

function h_rates(V)
    return (alpha_h(V), beta_h(V))
end





function hodgkin_huxley_rates_oop(u,p,t)
    V,n,m,h = u

    alpha_n_val, beta_n_val = n_rates(V)
    alpha_m_val, beta_m_val = m_rates(V)
    alpha_h_val, beta_h_val = h_rates(V)

    [(alpha_n_val, beta_n_val), (alpha_m_val, beta_m_val), (alpha_h_val, beta_h_val)]
end

function hodgkin_huxley_voltage_oop(u,p,t)
    V, n, m, h = u
    gNa, gK, gL, ENa, EK, EL, C = p

    INa = gNa * m^3 * h * (V - ENa)
    IK = gK * n^4 * (V - EK)
    IL = gL * (V - EL)

    [-(INa + IK + IL) / C]
end


p = [120.0, 36.0, 0.3, 50.0, -77.0, -54.4, 1.0]

# Initial conditions: V, n, m, h
u0 = [-65.0, 0.317, 0.05, 0.6]

# Add external current to the model
function hh_voltage_with_stimulus_oop(u, p, t)
    I_ext = 10.0
    du = hodgkin_huxley_voltage(u, p, t)
    du += [I_ext / p[7]]  # Add external current to dV/dt
    du
end

lars_funct_oop = RushLarsenFunction(hodgkin_huxley_rates_oop, hh_voltage_with_stimulus_oop, 2:4, 1:1)

# Solve the ODE system
tspan = (0.0, 1000.0)
prob = ODEProblem{false}(lars_funct, u0, tspan, p)

inter = init(prob, RushLarsen(), dt = 0.01)

step!(inter)

@btime solve(prob, RushLarsen(), dt=0.01)

@btime euler_sol = solve(prob, SimpleEuler(), dt = 0.01);

plot_simulation(sol)



function hodgkin_huxley_oop(u, p, t)
    V, n, m, h = u
    gNa, gK, gL, ENa, EK, EL, C = p

    # Get alpha and beta values from the rate functions
    alpha_n_val, beta_n_val = n_rates(V)
    alpha_m_val, beta_m_val = m_rates(V)
    alpha_h_val, beta_h_val = h_rates(V)

    # Compute ionic currents
    INa = gNa * m^3 * h * (V - ENa)
    IK = gK * n^4 * (V - EK)
    IL = gL * (V - EL)

    # Compute derivatives
    # du[1] = -(INa + IK + IL) / C  # dV/dt
    # du[2] = alpha_n_val * (1 - n) - beta_n_val * n  # dn/dt
    # du[3] = alpha_m_val * (1 - m) - beta_m_val * m  # dm/dt
    # du[4] = alpha_h_val * (1 - h) - beta_h_val * h  # dh/dt
    du = [-(INa + IK + IL) / C, alpha_n_val * (1 - n) - beta_n_val * n, alpha_m_val * (1 - m) - beta_m_val * m, alpha_h_val * (1 - h) - beta_h_val * h]
end


tspan = (0.0, 1000.0)
p = [120.0, 36.0, 0.3, 50.0, -77.0, -54.4, 1.0]

# Initial conditions: V, n, m, h
u0 = [-65.0, 0.317, 0.05, 0.6]

# Add external current to the model
function hh_with_stimulus_oop(u, p, t)
    I_ext = 10.0
    du = hodgkin_huxley_oop(u, p, t)
    du[1] += I_ext / p[7]  # Add external current to dV/dt
    du
end

hh_with_stimulus(u0, p, 0.0)[2]
# Solve the ODE system
prob = ODEProblem(hh_with_stimulus, u0, tspan, p)

@btime solve(prob, SimpleEuler(), dt = 0.01)






#======================================================================================
IIP
======================================================================================#

function hodgkin_huxley_rates_iip!(du, u, p, t)
    V, n, m, h = u

    alpha_n_val, beta_n_val = n_rates(V)
    alpha_m_val, beta_m_val = m_rates(V)
    alpha_h_val, beta_h_val = h_rates(V)

    du[1] = (alpha_n_val, beta_n_val)
    du[2] = (alpha_m_val, beta_m_val)
    du[3] = (alpha_h_val, beta_h_val)
end

function hodgkin_huxley_voltage_iip!(du, u, p, t)
    V, n, m, h = u
    gNa, gK, gL, ENa, EK, EL, C = p

    INa = gNa * m^3 * h * (V - ENa)
    IK = gK * n^4 * (V - EK)
    IL = gL * (V - EL)

    du[1] = -(INa + IK + IL) / C
end

function hh_voltage_with_stimulus_iip!(du, u, p, t)
    I_ext = 10.0
    hodgkin_huxley_voltage_iip!(du, u, p, t)
    du[1] += I_ext / p[7]  # Add external current to dV/dt
end

lars_funct_iip = RushLarsenFunction(hodgkin_huxley_rates_iip!, hh_voltage_with_stimulus_iip!, 2:4, 1:1)

p = [120.0, 36.0, 0.3, 50.0, -77.0, -54.4, 1.0]

# Initial conditions: V, n, m, h
u0 = [-65.0, 0.317, 0.05, 0.6]

tspan = (0.0, 50.0)

prob = ODEProblem{true}(lars_funct_iip, u0, tspan, p)

solve(prob, RushLarsen(), dt = 0.01)

integ = init(prob, RushLarsen(), dt = 0.01)

@allocated step!(integ)

@btime 

@profview for i in 1:10000000 step!(integ) end

@btime sol = solve(prob, RushLarsen(), dt = 0.01);

plot_simulation(sol)


function hodgkin_huxley_iip(du,u, p, t)
    V, n, m, h = u
    gNa, gK, gL, ENa, EK, EL, C = p

    # Get alpha and beta values from the rate functions
    alpha_n_val, beta_n_val = n_rates(V)
    alpha_m_val, beta_m_val = m_rates(V)
    alpha_h_val, beta_h_val = h_rates(V)

    # Compute ionic currents
    INa = gNa * m^3 * h * (V - ENa)
    IK = gK * n^4 * (V - EK)
    IL = gL * (V - EL)

    # Compute derivatives
    du[1] = -(INa + IK + IL) / C  # dV/dt
    du[2] = alpha_n_val * (1 - n) - beta_n_val * n  # dn/dt
    du[3] = alpha_m_val * (1 - m) - beta_m_val * m  # dm/dt
    du[4] = alpha_h_val * (1 - h) - beta_h_val * h  # dh/dt
end

function hh_with_stimulus_iip(du, u, p, t)
    I_ext = 10.0
    hodgkin_huxley_iip(du, u, p, t)
    du[1] = du[1] + I_ext / p[7]  # Add external current to dV/dt
end

p = [120.0, 36.0, 0.3, 50.0, -77.0, -54.4, 1.0]

# Initial conditions: V, n, m, h
u0 = [-65.0, 0.317, 0.05, 0.6]

tspan = (0.0, 50.0)

prob = ODEProblem(hh_with_stimulus_iip, u0, tspan, p)

reg_int = init(prob, SimpleEuler(), dt = 0.01)

@btime step!(reg_int)

@profview_allocs step!(reg_int) sample_rate = 1.0

@btime solve(prob, SimpleEuler(), dt=0.01);


function plot_simulation(sol)
    p1 = plot(sol, vars=(0, 1), xlabel="Time (ms)", ylabel="Membrane Potential (mV)",
        label="V", linewidth=2, title="Hodgkin-Huxley Model")

    p2 = plot(sol, vars=(0, 2:4), xlabel="Time (ms)", ylabel="Gating Variables",
        label=["n" "m" "h"], linewidth=2, title="Gating Variables")

    return plot(p1, p2, layout=(2, 1), size=(800, 600))
end

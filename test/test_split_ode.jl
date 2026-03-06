using Test
using RushLarsenSolvers
using OrdinaryDiffEq
using BlockDiagonals

@testset "Split ODE Form" begin
    # Create a simple test GRLFunction
    # du/dt = a*u + b where a and b are simple functions

    @testset "Simple scalar case" begin
        # a(u,p,t) = -p[1], b(u,p,t) = p[2]
        # Solution: u(t) = (u0 - b/a)*exp(a*t) + b/a
        a_func(u, p, t) = reshape([-p[1]], 1, 1)
        b_func(u, p, t) = [p[2]]

        grl_f = GRLFunction(a_func, b_func)

        u0 = [1.0]
        tspan = (0.0, 1.0)
        p = [2.0, 1.0]  # a = -2, b = 1

        # Analytical solution: u(t) = (1 - 0.5)*exp(-2*t) + 0.5
        u_exact(t) = 0.5 * exp(-2*t) + 0.5

        # Test out-of-place
        split_prob = split_ode_problem(grl_f, u0, tspan, p)
        sol = solve(split_prob, ETDRK2(), dt=0.01)

        @test sol[end][1] ≈ u_exact(1.0) rtol=1e-3

        # Test in-place
        split_prob_iip = split_ode_problem_iip(grl_f, u0, tspan, p)
        sol_iip = solve(split_prob_iip, ETDRK2(), dt=0.01)

        @test sol_iip[end][1] ≈ u_exact(1.0) rtol=1e-3
    end

    @testset "Vector case with BlockDiagonal" begin
        # System with 2 independent gates
        # du1/dt = -a1*u1 + b1
        # du2/dt = -a2*u2 + b2

        function a_func(u, p, t)
            a1, a2 = p[1], p[2]
            return BlockDiagonal([
                reshape([-a1], 1, 1),
                reshape([-a2], 1, 1)
            ])
        end

        function b_func(u, p, t)
            b1, b2 = p[3], p[4]
            return [b1, b2]
        end

        grl_f = GRLFunction(a_func, b_func)

        u0 = [0.5, 0.3]
        tspan = (0.0, 1.0)
        p = [1.0, 2.0, 0.2, 0.4]  # a1=1, a2=2, b1=0.2, b2=0.4

        # Analytical solutions
        u1_exact(t) = (0.5 - 0.2) * exp(-1.0*t) + 0.2
        u2_exact(t) = (0.3 - 0.4/2) * exp(-2.0*t) + 0.2

        split_prob = split_ode_problem(grl_f, u0, tspan, p)
        sol = solve(split_prob, ETDRK4(), dt=0.01)

        @test sol[end][1] ≈ u1_exact(1.0) rtol=1e-3
        @test sol[end][2] ≈ u2_exact(1.0) rtol=1e-3
    end

    @testset "State-dependent coefficients" begin
        # du/dt = -u^2 * u + 1
        # Here a(u,p,t) = -u^2 (state-dependent)

        function a_func(u, p, t)
            return reshape([-u[1]^2], 1, 1)
        end

        function b_func(u, p, t)
            return [1.0]
        end

        grl_f = GRLFunction(a_func, b_func)

        u0 = [0.5]
        tspan = (0.0, 0.5)
        p = nothing

        split_prob = split_ode_problem(grl_f, u0, tspan, p)

        # Should solve without error
        sol = solve(split_prob, ETDRK2(), dt=0.01)

        @test length(sol.t) > 0
        @test all(isfinite, sol[end])
    end

    @testset "Compatibility with different exponential methods" begin
        # Test that the split form works with various exponential integrators

        a_func(u, p, t) = reshape([-1.0], 1, 1)
        b_func(u, p, t) = [0.5]
        grl_f = GRLFunction(a_func, b_func)

        u0 = [1.0]
        tspan = (0.0, 1.0)
        p = nothing

        split_prob = split_ode_problem(grl_f, u0, tspan, p)

        # Test various methods
        methods = [
            ETDRK2(),
            ETDRK4(),
            Exp4(),
            EPIRK4s3A(),
            EPIRK5P1(),
        ]

        for alg in methods
            sol = solve(split_prob, alg, dt=0.1)
            @test length(sol.t) > 0
            @test all(isfinite, sol[end])
        end
    end

    @testset "Matrix operator extraction" begin
        a_func(u, p, t) = [1.0 2.0; 3.0 4.0]
        b_func(u, p, t) = [1.0, 2.0]

        grl_f = GRLFunction(a_func, b_func)

        A_op = matrix_operator(grl_f)

        u = [1.0, 1.0]
        p = nothing
        t = 0.0

        A = A_op(u, p, t)

        @test A ≈ [1.0 2.0; 3.0 4.0]
    end
end

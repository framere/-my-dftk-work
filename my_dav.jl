using LinearAlgebra
using Printf
using JLD2
using IterativeSolvers
using LinearMaps
using DataStructures
using Statistics

mutable struct EVHistory
    λ::Float64
    res::Vector{Float64}
end

function match_eigenvalue!(histories::Vector{EVHistory}, λ_new::Float64; tol=1e-2)
    best_idx = nothing
    best_err = Inf

    for (i, h) in enumerate(histories)
        err = abs(λ_new - h.λ)/max(abs(λ_new), abs(h.λ))
        if err < best_err
            best_err = err
            best_idx = i
        end
    end

    if best_idx !== nothing && best_err < tol
        histories[best_idx].λ = λ_new
        return best_idx
    end

    push!(histories, EVHistory(λ_new, Float64[]))
    return length(histories)
end

function correction_equations_minres(A, U, lambdas, R; tol=1e-1, maxiter=100)
    n, k = size(U)
    T_elem = eltype(U)
    S = zeros(T_elem, n, k)
    total_iter = 0

    for j in 1:k
        λ, r = lambdas[j], R[:, j]

        M_apply = function(x)
            x_perp = x - (U * (U' * x))
            x_perp_mat = reshape(x_perp, :, 1)
            tmp_mat = (A * x_perp_mat) - λ * x_perp_mat
            tmp = vec(tmp_mat)
            res = tmp - (U * (U' * tmp))
            return res
        end

        M_op = LinearMap{T_elem}(M_apply, n, n; ishermitian=true)

        rhs = r - (U * (U' * r))
        rhs = -rhs

        s_j, msg = minres(M_op, rhs; reltol=tol, maxiter=maxiter, log=true)
        m = match(r"(\d+)\s+iterations", string(msg))
        if m !== nothing
            niter = parse(Int, m.captures[1])
            total_iter += niter
        else
            println("No iteration number found in message: ", msg)
        end

        s_j = s_j - (U * (U' * s_j))
        S[:, j] = s_j
    end
    println("Total MINRES iterations: ", total_iter)
    return S
end

function select_corrections_ORTHO(t_candidates, V, V_lock, η, droptol; maxorth=2)
    n, ν = size(t_candidates)
    n_b = 0
    T_hat = Matrix{eltype(t_candidates)}(undef, n, ν)

    if size(V_lock,2) > 0
        W = hcat(V, V_lock)
    else
        W = V
    end
    have_W = size(W,2) > 0

    for i in 1:ν
        t_i = @view t_candidates[:, i]
        old_norm = norm(t_i)
        if old_norm < droptol
            continue
        end
        t = copy(t_i)

        for _ in 1:maxorth
            if have_W
                coeffs = W' * t
                t .-= W * coeffs
            end
            new_norm = norm(t)
            if new_norm > η * old_norm
                old_norm = new_norm
                break
            end
            old_norm = new_norm
        end

        final_norm = norm(t)
        if final_norm < droptol
            continue
        end
        n_b += 1
        T_hat[:, n_b] = t / final_norm
    end

    return T_hat[:,1:n_b], n_b
end

function degeneracy_detector(vals::AbstractVector{T}; tol=5e-3) where T<:Number
    inds = sortperm(vals)
    sorted_vals = vals[inds]
    groups = Vector{Vector{Int}}()
    current_group = [inds[1]]
    center = sorted_vals[1]

    for k in 2:length(vals)
        v = sorted_vals[k]
        rel_dist = abs(v - center) / max(abs(v), abs(center))
        if rel_dist < tol
            push!(current_group, inds[k])
            center = mean(vals[current_group])
        else
            if length(current_group) > 1
                push!(groups, copy(current_group))
            end
            current_group = [inds[k]]
            center = v
        end
    end

    if length(current_group) > 1
        push!(groups, current_group)
    end
    return groups
end

function is_stagnating(hist::Vector{Float64}; tol=0.1, window=2)
    length(hist) < window && return false
    r_old, r_new = hist[end-window+1], hist[end]
    if r_old == 0.0
        return r_new == 0.0
    end
    return abs(r_old - r_new) / r_old < tol
end

function my_davidson(
    A,
    D_kin::AbstractVector{<:Real},
    V::AbstractMatrix{T},
    n_aux::Integer,
    l::Integer,
    thresh::Float64,
    max_iter::Integer;
    use_jd::Bool = true          # set false to disable Jacobi-Davidson / stagnation logic entirely
)::Tuple{Vector{Float64}, Matrix{T}} where {T <: Complex}

    n = size(V, 1)
    n_b = size(V, 2)
    l_buffer = max(1, round(Int, l * 1.75))
    nu_0 = max(l_buffer, n_b)
    nevf = 0
    Nlow = size(V, 2)

    basis = A.base_op.basis
    kpt = A.base_op.kpoint

    println("Starting Davidson with n_aux = $n_aux, l_buffer = $l_buffer, l = $l, thresh = $thresh, max_iter = $max_iter, use_jd = $use_jd")

    Eigenvalues = Float64[]
    Ritz_vecs = Matrix{T}(undef, n, 0)
    V_lock = Matrix{T}(undef, n, 0)

    # Residual histories are only needed for stagnation detection (JD mode)
    residual_histories = EVHistory[]
    converged_cooldown = DefaultDict{Int, Int}(() -> 0)
    cooldown_iters = 1

    iter = 0

    if size(V,2) == 0
        error("Initial subspace V must have at least one column.")
    end

    while nevf < l_buffer
        iter += 1
        n_c = 0
        if iter > max_iter
            println("Max iterations ($max_iter) reached without full expected convergence. Returning what we have.")
            return (Eigenvalues, Ritz_vecs)
        end

        # Orthogonalize against locked vectors
        if size(V_lock, 2) > 0
            V_lock = Matrix(qr(V_lock).Q)
            for i in 1:size(V_lock, 2)
                v_lock = V_lock[:, i]
                for j in 1:size(V, 2)
                    V[:, j] .-= v_lock * (v_lock' * V[:, j])
                end
            end
        end

        V = Matrix(qr(V).Q)

        # Rayleigh-Ritz
        AV = A * V
        H = Hermitian(V' * AV)
        nu = min(size(H, 2), nu_0 - nevf)
        Σ, U = eigen(H, 1:nu)
        X = V * U
        AX = A * X
        R = X .* Σ' .- AX
        norms = vec(norm.(eachcol(R)))

        sorted_indices = sortperm(Σ)
        Σ_sorted = Σ[sorted_indices]
        X_sorted = X[:, sorted_indices]
        R_sorted = R[:, sorted_indices]
        norms_sorted = norms[sorted_indices]

        # Update residual histories only when JD is enabled (needed for stagnation detection)
        if use_jd
            for (local_idx, rnorm) in enumerate(norms_sorted)
                λ = Σ_sorted[local_idx]
                idx = match_eigenvalue!(residual_histories, λ; tol=1e-3)
                push!(residual_histories[idx].res, rnorm)
                if length(residual_histories[idx].res) > 4
                    popfirst!(residual_histories[idx].res)
                end
            end
        end

        current_cutoff = min(l - nevf, length(Σ_sorted))
        deg_groups = degeneracy_detector(Σ_sorted; tol = 1e-3)
        locked_sorted_positions = Int[]

        # Lock degenerate clusters
        for group in deg_groups
            inside = filter(i -> i <= current_cutoff, group)
            outside = filter(i -> i > current_cutoff, group)
            res_ok_inside = all(norms_sorted[i] < thresh for i in inside)
            res_ok = all(norms_sorted[i] < thresh for i in group)

            if isempty(inside)
                continue
            elseif isempty(outside)
                if res_ok
                    for gi in group
                        converged_cooldown[gi] += 1
                    end
                    if all(converged_cooldown[gi] ≥ cooldown_iters for gi in group)
                        for gi in group
                            λ = Σ_sorted[gi]; xvec = X_sorted[:, gi]
                            push!(Eigenvalues, float(λ))
                            Ritz_vecs = hcat(Ritz_vecs, xvec)
                            V_lock = hcat(V_lock, xvec)
                            push!(locked_sorted_positions, gi)
                            nevf += 1; n_c += 1
                            if use_jd
                                hist_idx = match_eigenvalue!(residual_histories, λ; tol=1e-3)
                                deleteat!(residual_histories, hist_idx)
                            end
                        end
                    end
                end
                continue
            else
                if res_ok_inside
                    for gi in inside
                        converged_cooldown[gi] += 1
                    end
                    if all(converged_cooldown[gi] ≥ cooldown_iters for gi in inside)
                        for gi in inside
                            λ = Σ_sorted[gi]; xvec = X_sorted[:, gi]
                            push!(Eigenvalues, float(λ))
                            Ritz_vecs = hcat(Ritz_vecs, xvec)
                            V_lock = hcat(V_lock, xvec)
                            push!(locked_sorted_positions, gi)
                            nevf += 1; n_c += 1
                            if use_jd
                                hist_idx = match_eigenvalue!(residual_histories, λ; tol=1e-3)
                                deleteat!(residual_histories, hist_idx)
                            end
                        end
                    end
                end
            end
        end

        # Lock isolated eigenvalues
        for i in 1:current_cutoff
            if i in locked_sorted_positions
                continue
            end
            if norms_sorted[i] < thresh
                converged_cooldown[i] += 1
                if converged_cooldown[i] ≥ cooldown_iters
                    λ = Σ_sorted[i]; xvec = X_sorted[:, i]
                    push!(Eigenvalues, float(λ))
                    Ritz_vecs = hcat(Ritz_vecs, xvec)
                    V_lock = hcat(V_lock, xvec)
                    push!(locked_sorted_positions, i)
                    nevf += 1; n_c += 1
                    if use_jd
                        hist_idx = match_eigenvalue!(residual_histories, λ; tol=1e-3)
                        deleteat!(residual_histories, hist_idx)
                    end
                end
            end
        end

        if nevf >= l
            println("Converged all required eigenvalues (cluster-aware). Iter = $iter")
            return (Eigenvalues, Ritz_vecs)
        end

        # Prepare candidates for corrections
        all_sorted_positions = collect(1:length(Σ_sorted))
        non_conv_positions = setdiff(all_sorted_positions, locked_sorted_positions)
        non_conv_sorted_positions = sort(non_conv_positions, by=i->Σ_sorted[i])
        nbuffer = max(1, l_buffer - nevf)
        keep_positions = non_conv_sorted_positions[1:min(length(non_conv_sorted_positions), nbuffer)]
        X_nc = X_sorted[:, keep_positions]
        Σ_nc = Σ_sorted[keep_positions]
        R_nc = R_sorted[:, keep_positions]

        # ----------------------------------------------------------------
        # Correction vectors
        # ----------------------------------------------------------------
        t = zeros(eltype(V), n, length(keep_positions))

        if !use_jd || iter < 9
            # Pure Davidson preconditioner for every vector
            for (i_local, _) in enumerate(keep_positions)
                R_real = ifft(basis, kpt, R_nc[:, i_local])   # 3D array (nx,ny,nz)
                C = -1.0 ./ (D_kin .- Σ_nc[i_local])          # flat vector, length Nfull
                t[:, i_local] = fft(basis, kpt, C .* vec(R_real))  # vec() flattens R_real
            end
        else
            # Hybrid Davidson / Jacobi-Davidson (stagnation-triggered)
            dav_indices = Int[]
            jd_indices  = Int[]

            for (i_local, pos) in enumerate(keep_positions)
                if pos > current_cutoff
                    push!(dav_indices, i_local)
                else
                    λ = Σ_sorted[pos]
                    hist_idx = match_eigenvalue!(residual_histories, λ; tol=1e-3)
                    hist = residual_histories[hist_idx].res
                    if length(hist) ≥ 2 && is_stagnating(hist; tol=0.1, window=2)
                        push!(jd_indices, i_local)
                    else
                        push!(dav_indices, i_local)
                    end
                end
            end
            println("Using Davidson for $(length(dav_indices)) vectors, JD for $(length(jd_indices)) vectors.")

            t_dav = zeros(eltype(V), n, length(dav_indices))
            for (j, i_local) in enumerate(dav_indices)
                R_real = ifft(basis, kpt, R_nc[:, i_local])
                C = -1.0 ./ (D_kin .- Σ_nc[i_local])
                t_dav[:, j] = fft(basis, kpt, C .* vec(R_real))  # ← add vec() here too
            end

            t_jd = if !isempty(jd_indices)
                correction_equations_minres(
                    A, X_nc[:, jd_indices], Σ_nc[jd_indices], R_nc[:, jd_indices];
                    tol=1e-1, maxiter=25
                )
            else
                zeros(eltype(V), n, 0)
            end

            t = hcat(t_dav, t_jd)
        end
        # ----------------------------------------------------------------

        i_max = argmin(Σ_sorted)
        println("Iter $iter: V_size = $n_b, Converged = $nevf, ‖r‖ (largest λ) = $(norms_sorted[i_max])")

        T_hat, n_b_hat = select_corrections_ORTHO(t, V, V_lock, 0.1, 1e-12)

        if size(V, 2) + n_b_hat > n_aux || n_b_hat == 0
            # Subspace too large or no new directions: restart from non-converged Ritz vectors
            max_new_size = n_aux - size(X_nc, 2)
            T_hat = T_hat[:, 1:min(n_b_hat, max(0, max_new_size))]
            V = hcat(X_nc, T_hat)
        else
            # Room to expand: just append the new correction vectors
            V = hcat(V, T_hat)
        end
    end

    return (Eigenvalues, Ritz_vecs)
end

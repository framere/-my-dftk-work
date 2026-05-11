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

# Match Ritz value to existing history (value-based, fuzzy)
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
        histories[best_idx].λ = λ_new   # update stored value
        return best_idx
    end

    # Otherwise create a new entry
    push!(histories, EVHistory(λ_new, Float64[]))
    return length(histories)
end

# === Global FLOP counter and helpers ===
global NFLOPs = 0

include("../../MA_best/FLOP_count.jl")

function correction_equations_minres(A, U, lambdas, R; tol=1e-1, maxiter=100)
    global NFLOPs
    n, k = size(U)
    T_elem = eltype(U)
    S = zeros(T_elem, n, k)
    total_iter = 0

    for j in 1:k
        λ, r = lambdas[j], R[:, j]

        M_apply = function(x)
            x_perp = x - (U * (U' * x)); 
            count_matmul_flops(k,1,n); count_matmul_flops(n,1,k); count_vec_add_flops(n)

            # Reshape to matrix for A multiplication
            x_perp_mat = reshape(x_perp, :, 1)
            tmp_mat = (A * x_perp_mat) - λ * x_perp_mat
            tmp = vec(tmp_mat)  # Convert back to vector
            
            count_matmul_flops(n,1,n); count_vec_scaling_flops(n); count_vec_add_flops(n)

            res = tmp - (U * (U' * tmp)); 
            count_matmul_flops(k,1,n); count_matmul_flops(n,1,k); count_vec_add_flops(n)
            return res
        end

        M_op = LinearMap{T_elem}(M_apply, n, n; ishermitian=true)

        rhs = r - (U * (U' * r)); 
        count_matmul_flops(k,1,n); count_matmul_flops(n,1,k); count_vec_add_flops(n)
        rhs = -rhs; count_vec_scaling_flops(n)

        s_j, msg = minres(M_op, rhs; reltol=tol, maxiter=maxiter, log=true)
        m = match(r"(\d+)\s+iterations", string(msg))
        if m !== nothing
            niter = parse(Int, m.captures[1])
            total_iter += niter
        else
            println("No iteration number found in message: ", msg)
        end

        s_j = s_j - (U * (U' * s_j)); 
        count_matmul_flops(k,1,n); count_matmul_flops(n,1,k); count_vec_add_flops(n)

        S[:, j] = s_j
    end
    println("Total MINRES iterations: ", total_iter)
    NFLOPs += total_iter * (2*n^2 + 4*n*k)  
    return S
end

function select_corrections_ORTHO(t_candidates, V, V_lock, η, droptol; maxorth=2)
    n, ν = size(t_candidates)
    n_b = 0

    # Preallocate maximum possible output (we trim at the end)
    T_hat = Matrix{eltype(t_candidates)}(undef, n, ν)

    # Build one combined orthogonalization basis
    if size(V_lock,2) > 0
        W = hcat(V, V_lock)
    else
        W = V
    end

    have_W = size(W,2) > 0   # for a cheap check

    for i in 1:ν
        t_i = @view t_candidates[:, i]

        # Early norm (avoids unnecessary work)
        old_norm = norm(t_i)
        count_norm_flops(length(t_i))
        if old_norm < droptol
            continue
        end

        # Copy so we don't mutate original
        t = copy(t_i)

        # === Orthogonalize t against W (block Gram-Schmidt)

        # We allow up to maxorth reorthogonalizations
        for _ in 1:maxorth
            if have_W
                # coeffs = W' * t
                count_matmul_flops(size(W,2), 1, n)
                coeffs = W' * t
                # t .-= W * coeffs
                count_matmul_flops(n, 1, size(W,2))
                t .-= W * coeffs
                count_vec_add_flops(n)
            end

            new_norm = norm(t)
            count_norm_flops(length(t))
            if new_norm > η * old_norm
                old_norm = new_norm
                break
            end
            old_norm = new_norm
        end

        # Final norm check
        final_norm = norm(t)
        count_norm_flops(length(t))
        if final_norm < droptol
            continue
        end

        # Normalize and accept
        n_b += 1
        # division by scalar
        count_vec_scaling_flops(n)
        T_hat[:, n_b] = t / final_norm
    end

    return T_hat[:,1:n_b], n_b
end

function degeneracy_detector(vals::AbstractVector{T}; tol=5e-3) where T<:Number
    # Sort eigenvalues but keep original index
    inds = sortperm(vals)
    sorted_vals = vals[inds]

    groups = Vector{Vector{Int}}()
    current_group = [inds[1]]
    center = sorted_vals[1]     # cluster center (updated mean)

    for k in 2:length(vals)
        v = sorted_vals[k]
        rel_dist = abs(v - center) / max(abs(v), abs(center))

        if rel_dist < tol
            # Add to current cluster
            push!(current_group, inds[k])

            # Update center (running mean)
            center = mean(vals[current_group])

        else
            # Finalize old group if size > 1
            if length(current_group) > 1
                push!(groups, copy(current_group))
            end

            # Start new cluster
            current_group = [inds[k]]
            center = v
        end
    end

    # Handle last group
    if length(current_group) > 1
        push!(groups, current_group)
    end

    return groups
end


# --- small helper 
function is_stagnating(hist::Vector{Float64}; tol=0.1, window=2)
    length(hist) < window && return false
    r_old, r_new = hist[end-window+1], hist[end]
    # avoid division by zero
    if r_old == 0.0
        return r_new == 0.0
    end
    return abs(r_old - r_new) / r_old < tol
end

function davidson(
    A::ProjectedShiftedOperator,
    D_real::AbstractArray{<:Real, 3},
    V::AbstractMatrix{T},
    eigenvectors_pool::AbstractMatrix{T},
    n_aux::Integer,
    l::Integer,
    thresh::Float64,
    max_iter::Integer,
    all_idxs::Vector{Int}
)::Tuple{Vector{Float64}, Matrix{T}} where {T <: Complex}

    n = size(V, 1)
    n_b = size(V, 2)
    l_buffer = max(1, round(Int, l * 1.75))
    nu_0 = max(l_buffer, n_b)
    nevf = 0
    Nlow = size(V, 2)

    # for FFTs get the basis and kpoint from the DFTK operator
    basis = A.base_op.basis
    kpt = A.base_op.kpoint

    println("Starting Davidson with n_aux = $n_aux, l_buffer = $l_buffer, l = $l, thresh = $thresh, max_iter = $max_iter")

    Eigenvalues = Float64[]
    Ritz_vecs = Matrix{T}(undef, n, 0)
    V_lock = Matrix{T}(undef, n, 0)


    residual_histories = EVHistory[]
    # Track convergence cool-down for each sorted position
    converged_cooldown = DefaultDict{Int, Int}(() -> 0)
    cooldown_iters = 1   # keep converged eigenvalues/groups for 3 extra iterations

    iter = 0

    # Ensure V is full rank / orthonormal initially
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
            count_qr_flops(n, size(V_lock,2))
            for i in 1:size(V_lock, 2)
                v_lock = V_lock[:, i]
                for j in 1:size(V, 2)
                    count_dot_product_flops(n)
                    count_vec_scaling_flops(n)
                    count_vec_add_flops(n)
                    V[:, j] .-= v_lock * (v_lock' * V[:, j])
                end
            end
        end

        # Orthonormalize the basis
        V = Matrix(qr(V).Q)
        count_qr_flops(n, size(V,2))

        # Rayleigh-Ritz
        AV = A * V
        count_matmul_flops(n, size(V,2), n)
        H = Hermitian(V' * AV)
        count_matmul_flops(size(V,2), size(AV,2), n)

        nu = min(size(H, 2), nu_0 - nevf)
        count_diag_flops(nu)
        Σ, U = eigen(H, 1:nu)
        X = V * U
        count_matmul_flops(n, nu, size(V,2))

        AX = A * X
        count_matmul_flops(n, nu, n)
        for col in 1:nu
            count_vec_scaling_flops(n)
        end
        R = X .* Σ' .- AX
        for col in 1:nu
            count_vec_add_flops(n)
        end

        norms = vec(norm.(eachcol(R)))
        for _ in 1:nu
            count_norm_flops(n)
        end

        # Sort Ritz values ascending
        sorted_indices = sortperm(Σ)
        Σ_sorted = Σ[sorted_indices]
        X_sorted = X[:, sorted_indices]
        R_sorted = R[:, sorted_indices]
        norms_sorted = norms[sorted_indices]

        # --- Update value-based residual histories ---
        for (local_idx, rnorm) in enumerate(norms_sorted)
            λ = Σ_sorted[local_idx]

            # find or create the history entry for this eigenvalue
            idx = match_eigenvalue!(residual_histories, λ; tol=1e-3)

            # append new residual
            push!(residual_histories[idx].res, rnorm)

            # keep buffer short
            if length(residual_histories[idx].res) > 4
                popfirst!(residual_histories[idx].res)
            end
        end

        current_cutoff = min(l - nevf, length(Σ_sorted))
        deg_groups = degeneracy_detector(Σ_sorted; tol = 1e-3)
        locked_sorted_positions = Int[]

        # --- Lock degenerate clusters ---
        for group in deg_groups
            inside = filter(i -> i <= current_cutoff, group)
            outside = filter(i -> i > current_cutoff, group)
            res_ok_inside = all(
                norms_sorted[i] < (thresh)
                for i in inside
            )

            res_ok = all(
                norms_sorted[i] < (thresh)
                for i in group
            )

            if isempty(inside)
                continue
            elseif isempty(outside)
                if res_ok
                    # increase cooldown counter for the whole group
                    for gi in group
                        converged_cooldown[gi] += 1
                    end

                    if all(converged_cooldown[gi] ≥ cooldown_iters for gi in group)
                        # lock only if stayed converged for >= cooldown_iters
                        for gi in group
                            λ = Σ_sorted[gi]; xvec = X_sorted[:, gi]
                            push!(Eigenvalues, float(λ))
                            Ritz_vecs = hcat(Ritz_vecs, xvec)
                            V_lock = hcat(V_lock, xvec)
                            push!(locked_sorted_positions, gi)
                            nevf += 1; n_c += 1
                            hist_idx = match_eigenvalue!(residual_histories, λ; tol=1e-3)
                            deleteat!(residual_histories, hist_idx)
                        end
                    end
                end
                continue
            else
                if res_ok_inside
                    # increase cool-down counter only for the inside part
                    for gi in inside
                        converged_cooldown[gi] += 1
                    end

                    # lock only if cooldown threshold reached
                    if all(converged_cooldown[gi] ≥ cooldown_iters for gi in inside)
                        for gi in inside
                            λ = Σ_sorted[gi]; xvec = X_sorted[:, gi]
                            push!(Eigenvalues, float(λ))
                            Ritz_vecs = hcat(Ritz_vecs, xvec)
                            V_lock = hcat(V_lock, xvec)
                            push!(locked_sorted_positions, gi)
                            nevf += 1; n_c += 1
                            hist_idx = match_eigenvalue!(residual_histories, λ; tol=1e-3)
                            deleteat!(residual_histories, hist_idx)
                        end
                    end
                end
            end
        end

        # --- Lock isolated eigenvalues ---
        for i in 1:current_cutoff
            if i in locked_sorted_positions
                continue
            end
            λ = Σ_sorted[i]
            adaptive_thresh = thresh
            if norms_sorted[i] < adaptive_thresh
                converged_cooldown[i] += 1   # increment cooldown counter

                if converged_cooldown[i] ≥ cooldown_iters
                    λ = Σ_sorted[i]; xvec = X_sorted[:, i]
                    push!(Eigenvalues, float(λ))
                    Ritz_vecs = hcat(Ritz_vecs, xvec)
                    V_lock = hcat(V_lock, xvec)
                    push!(locked_sorted_positions, i)
                    nevf += 1; n_c += 1
                    hist_idx = match_eigenvalue!(residual_histories, λ; tol=1e-3)
                    deleteat!(residual_histories, hist_idx)
                end
            end
        end

        if nevf >= l
            println("Converged all required eigenvalues (cluster-aware). Iter = $iter")
            return (Eigenvalues, Ritz_vecs)
        end

        # --- Prepare candidates for corrections ---
        all_sorted_positions = collect(1:length(Σ_sorted))
        non_conv_positions = setdiff(all_sorted_positions, locked_sorted_positions)
        non_conv_sorted_positions = sort(non_conv_positions, by=i->Σ_sorted[i])
        nbuffer = max(1, l_buffer - nevf)
        keep_positions = non_conv_sorted_positions[1:min(length(non_conv_sorted_positions), nbuffer)]
        X_nc = X_sorted[:, keep_positions]
        Σ_nc = Σ_sorted[keep_positions]
        R_nc = R_sorted[:, keep_positions]

        # --- Compute correction vectors ---
        ϵ = 1e-8
        t = zeros(eltype(V), n, length(keep_positions))

        if iter < 9
            for (i_local, pos) in enumerate(keep_positions)
                R_real = ifft(basis, kpt, R_nc[:, i_local]) # FFT to real space
                C = -1.0 ./ (D_real .- Σ_nc[i_local])
                t_real = C .* R_real # apply C
                t[:, i_local] = fft(basis, kpt, t_real) # FFT back to reciprocal space
                count_vec_scaling_flops(n)
            end
        else
            # Hybrid Davidson-JD
            dav_indices = Int[]
            jd_indices = Int[]

            for (i_local, pos) in enumerate(keep_positions)
                if pos > current_cutoff
                    # Anything beyond lc cutoff → always Davidson used for the buffer vectors. We don't spend so much time computing JD for high-lying states.
                    push!(dav_indices, i_local)
                else
                    λ = Σ_sorted[pos]

                    # get the history entry for this value
                    hist_idx = match_eigenvalue!(residual_histories, λ; tol=1e-3)
                    hist = residual_histories[hist_idx].res

                    stagnating = length(hist) ≥ 2 && is_stagnating(hist; tol=0.1, window=2)

                    if stagnating
                        push!(jd_indices, i_local)
                    else
                        push!(dav_indices, i_local)
                    end
                end
            end
            println("Using Davidson for $(length(dav_indices)) vectors, JD for $(length(jd_indices)) vectors.")

            # Davidson corrections
            t_dav = zeros(eltype(V), n, length(dav_indices))
            for (j, i_local) in enumerate(dav_indices)
                    R_real = ifft(basis, kpt, R_nc[:, i_local]) # FFT to real space
                    C = -1.0 ./ (D_real .- Σ_nc[i_local])
                    t_real = C .* R_real # apply C
                    t_dav[:, j] = fft(basis, kpt, t_real) # FFT back to reciprocal space   
            end

            # JD corrections
            if !isempty(jd_indices)
                X_jd = X_nc[:, jd_indices]
                Σ_jd = Σ_nc[jd_indices]
                R_jd = R_nc[:, jd_indices]
                t_jd = correction_equations_minres(A, X_jd, Σ_jd, R_jd; tol=1e-1, maxiter=25)
            else
                t_jd = zeros(eltype(V), n, 0)
            end

            t = hcat(t_dav, t_jd)
        end

        # Print iteration info
        i_max = argmin(Σ_sorted)
        norm_largest_Ritz = norms_sorted[i_max]
        println("Iter $iter: V_size = $n_b, Converged = $nevf, ‖r‖ (largest λ) = $norm_largest_Ritz")

        # --- Orthonormalize & update V ---
        T_hat, n_b_hat = select_corrections_ORTHO(t, V, V_lock, 0.1, 1e-12)

        # Update search space V
        if size(V, 2) + n_b_hat > n_aux && n_c > 0
            start_idx = Nlow + 1 + (nevf - n_c)
            end_idx = Nlow + nevf
            
            # Clamp to valid range
            start_idx = max(1, min(start_idx, length(all_idxs)))
            end_idx = max(start_idx, min(end_idx, length(all_idxs)))
            
            if start_idx <= end_idx && end_idx <= length(all_idxs)
                extra_idx = all_idxs[start_idx:end_idx]
                if size(X_nc, 2) == 0
                    println("Warning: X_nc is empty when rebuilding V, using only T_hat and extra vectors.")
                end
                max_new_size = n_aux - size(X_nc, 2) - length(extra_idx)
                T_hat = T_hat[:, 1:min(n_b_hat, max(0, max_new_size))]
                V = hcat(X_nc, T_hat, eigenvectors_pool[:, extra_idx])
            else
                # Pool exhausted
                max_new_size = n_aux - size(X_nc, 2)
                T_hat = T_hat[:, 1:min(n_b_hat, max(0, max_new_size))]
                V = hcat(X_nc, T_hat)
            end

        elseif size(V, 2) + n_b_hat > n_aux || n_b_hat == 0
            max_new_size = n_aux - size(X_nc, 2)
            T_hat = T_hat[:, 1:min(n_b_hat, max(0, max_new_size))]            
            V = hcat(X_nc, T_hat)

        elseif n_c > 0
            start_idx = Nlow + 1 + (nevf - n_c)
            end_idx = Nlow + nevf
            
            # Clamp to valid range
            start_idx = max(1, min(start_idx, length(all_idxs)))
            end_idx = max(start_idx, min(end_idx, length(all_idxs)))
            
            if start_idx <= end_idx && end_idx <= length(all_idxs)
                extra_idx = all_idxs[start_idx:end_idx]
                V = hcat(V, T_hat, eigenvectors_pool[:, extra_idx])
            else
                # Pool exhausted
                V = hcat(V, T_hat)
            end

        else
            V = hcat(V, T_hat)
        end

        n_b = size(V, 2)

        if size(V, 2) == 0
            println("Warning: V is empty, rebuilding from eigenvector pool.")
            start_idx = Nlow + 1 + (nevf - n_c)
            end_idx = Nlow + nevf + Nlow
            
            # Clamp to valid range
            start_idx = max(1, min(start_idx, length(all_idxs)))
            end_idx = max(start_idx, min(end_idx, length(all_idxs)))
            
            if start_idx <= end_idx && end_idx <= length(all_idxs)
                extra_idx = all_idxs[start_idx:end_idx]
                V = eigenvectors_pool[:, extra_idx]
            else
                error("Cannot rebuild V: eigenvector pool exhausted and no vectors available")
            end
        end
    end

    return (Eigenvalues, Ritz_vecs)
end


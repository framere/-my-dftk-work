
function sample_r_according_to_potential(V, n_samples)
    weights = abs.(copy(vec(V)))
    ngrid = length(weights)
    grid_size = size(V)
    selected_samples = CartesianIndex[]
    for _ in 1:n_samples
        idx = sample(1:ngrid, Weights(weights))
        sample_idx = CartesianIndices(grid_size)[idx]
        push!(selected_samples, sample_idx)
        weights[idx]=0.0 # make re-sampling impossible
    end
    return selected_samples
end


# put narrow Gaussians on selected points in space. Points are randomly 
# drawn according to the real space weight V.
function construct_Gaussians_according_to_potential(V, n_samples, basis, kpt, orbitalType)
    samples = sample_r_according_to_potential(V, n_samples) # generate samples
    ϕk = zeros(orbitalType, length(kpt.G_vectors), n_samples)
    α = basis.Ecut/8 # for narrow Gaussian (might need optimization)
    G_vecs = G_vectors(basis, kpt)
    for (a,r_idx) in enumerate(samples)
        r_frac = (Tuple(r_idx) .- 1) ./ basis.fft_size # from index to fractional coordinate
        r = basis.model.lattice * collect(r_frac) # from fractional coordinate to r
        for (iG, G_red) in enumerate(G_vecs)
            G_cart = basis.model.recip_lattice * (G_red + kpt.coordinate)
            G_norm_sq = dot(G_cart, G_cart)
            # build orbital centered at r with real space spread σ=1/2*sqrt(3/α)
            ϕk[iG,a] = exp(-G_norm_sq / (4α)) * cis(-dot(G_cart, r)) 
        end
        ϕk[:,a] ./= norm(ϕk[:,a]) # normalize
    end
    # orthogonalize initial
    qr_decomp = qr(ϕk)
    ϕk = Matrix(qr_decomp.Q)
end


function construct_stochastic_orbitals(N, kpt, orbitalType)
    NG = length(kpt.G_vectors)
    radius = rand(NG,N)
    phase = cis.(2π .* rand(NG,N))
    ϕk = zeros(orbitalType, length(kpt.G_vectors), N)
    ϕk = radius .* phase
    for a in 1:N
        ϕk[:,a] ./= norm(ϕk[:,a]) # normalize
    end
    # orthogonalize
    qr_decomp = qr(ϕk)
    ϕk = Matrix(qr_decomp.Q)
end


# simply return the plane-wave basis vectors with lowest kinetic energy
function construct_HEG_basis(N, kpt, basis, orbitalType)
    G_vecs_cart = [basis.model.recip_lattice * G for G in G_vectors(basis, kpt)]
    G_sq = sum.(abs2, G_vecs_cart)
    perm = sortperm(G_sq)
    lowest_G_indices = perm[1:N]
    ϕk = zeros(orbitalType, length(kpt.G_vectors), N)
    for (a, G_idx) in enumerate(lowest_G_indices)
        ϕk[G_idx, a] = 1.0
    end
    return ϕk
end



# perform a pivoted Cholesky decomposition of K(r,r')
# gives very good initial guess, but is also very expensive.
function construct_Cholesky_of_K(scfres::NamedTuple, basis::PlaneWaveBasis, ik::Integer, σ::Integer, N::Integer)
    ψocc, occupation_occ = DFTK.select_occupied_orbitals(basis, scfres.ψ, scfres.occupation; threshold=1e-8)
    occk = occupation_occ[ik]
    ψocck = ψocc[ik]
    Nocc = size(ψocck,2)

    # Fourier transform all occupied orbitals to real-space
    orbitalType = eltype(ψocc[1])
    kpt = basis.kpoints[ik]
    ψocck_real = zeros(orbitalType, basis.fft_size..., Nocc)
    for i = 1:Nocc
        ψocck_real[:,:,:,i] = ifft(basis, kpt, ψocck[:, i])
    end
    
    # filter our the ExactExchange term from our basis to perform a FFT of
    # poisson_green_coeffs=4π/G² which gives us 1/r on the grid respecting the MIC.
    ExactExchangeTerm = only([term for term in basis.terms if term isa DFTK.TermExactExchange])
    unit_cell_volume = abs(det(basis.model.lattice))
    CoulombPotential_real = irfft(basis, Complex.(ExactExchangeTerm.poisson_green_coeffs)) / sqrt(unit_cell_volume)
    v0 = CoulombPotential_real[1,1,1] # Coulomb potential at r=0

    total_iterations = Int(N*(N-1)/2)
    p = Progress(total_iterations; desc="Pivoted Cholesky decomposition of K(r,r')", dt=0.5, barlen=20, color=:black)

    # perform a Pivoted Cholesky decomposition of K(r,r') to get linearly independent guess vectors.
    D = copy(scfres.ρ[:,:,:,σ]) * v0 # get the diagonal K(r,r) exploiting ρ(r) = K(r,r)*v(0)
    ϕk_real = zeros(orbitalType, basis.fft_size..., N)
    for a=1:N
        pivot_val, r_a = findmax(D) 
        
        # build this column ϕ_a(r) =  K(r,r_a) 
        weights = occk .* conj(ψocck_real[r_a,:]) # <ψocck_i|r_a> * f_i
        ϕwork = ψocck * weights # <G|ϕwork> = sum_i <G|ψocck_i> * <ψocck_i|r_a> * f_i
        ϕwork = ifft(basis, kpt, ϕwork) # FFT to real space
        shift_ra = r_a - CartesianIndex(1,1,1) 
        ShiftedCoulombPotential_real = circshift(CoulombPotential_real, Tuple(shift_ra)) # = 1/|r-r_a|
        ϕwork .*= ShiftedCoulombPotential_real # ϕwork(r) = ρ(r,r_a)/|r-r_a|

        (a==1) && next!(p)
        for b=1:(a-1) # orthogonalize against previous Cholesky vectors
            ϕwork[:,:,:] .-= ϕk_real[:,:,:,b] * conj(ϕk_real[r_a,b])
            next!(p) # update Progress bar
        end
        ϕk_real[:,:,:,a] = ϕwork[:,:,:] ./ sqrt(ϕwork[r_a]) # Cholesky normalize
        @. D -= abs2(ϕk_real[:,:,:,a])  # update diagonal by subtracting |ϕ_a(r)|^2
    end

    # FFT back to reciprocal space
    ϕk = zeros(orbitalType, length(kpt.G_vectors), N)
    for a=1:N
        ϕk[:,a] = fft(basis, kpt, ϕk_real[:,:,:,a]) 
        ϕk[:,a] ./= norm(ϕk[:,a],2) # normalize
    end

    # orthogonalize Cholesky vectors
    qr_decomp = qr(ϕk)
    ϕk = Matrix(qr_decomp.Q)
    
    return ϕk
end



# a simple implementation of the block Davidson method 
function davidson_tobias(
    A::ProjectedShiftedOperator, # linear operator
    D_real::AbstractArray{<:Real, 3}, # precondition in real space
    V::AbstractMatrix{T},
    ψocck::AbstractMatrix{T}, # the occupied orbitals for the projector
    Naux::Integer,
    thresh::Float64
)::Tuple{Vector{T},Matrix{T}} where T<:Number
    
    Nlow = size(V,2) # number of eigenvalues we are interested in
    if Naux < Nlow
        println("ERROR: auxiliary basis must not be smaller than number of target eigenvalues")
    end

    # for FFTs get the basis and kpoint from the DFTK operator
    basis = A.base_op.basis
    kpt = A.base_op.kpoint

    # iterations
    iter = 0
    while true
        iter = iter + 1
        
        # orthogonalize guess orbitals (using QR decomposition)
        qr_decomp = qr(V)
        V = Matrix(qr_decomp.Q)

        # construct and diagonalize Rayleigh matrix
        H = V' * (A*V)     
        H = Hermitian(H) 
        Σ, U = eigen(H, 1:Nlow)

        X = V*U # Ritz vecors
        R = X.*Σ' - A*X
        Rnorm = norm(R,2) # Frobenius norm

        # status output
        output = @sprintf("iter=%6d  Rnorm=%11.3e  size(V,2)=%6d\n", iter, Rnorm, size(V,2))
        print(output)
        
        if Rnorm < thresh 
            println("converged!")
            return (Σ, X)
        end

        # DOESN'T HELP
        # update guess space using preconditioner 
        # t = zero(similar(R)) 
        # for i = 1:size(t,2)
        #   R_real = ifft(basis, kpt, R[:,i]) # FFT to real space
        #   C = -1.0 ./ (D_real .- Σ[i])
        #   t_real = C .* R_real # apply C
        #   t[:,i] = fft(basis, kpt, t_real) # FFT back to reciprocal space
        # end
        
        t = R # no preconditioner

        # update guess basis
        if size(V,2) <= Naux-Nlow
            V = hcat(V,t) # concatenate V and t
        else
            V = hcat(X,t) # concatenate X and t 
        end
    end
end
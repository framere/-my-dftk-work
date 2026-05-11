using Pkg
Pkg.activate(".")

using DFTK
using PseudoPotentialData
using LinearAlgebra
using StatsBase
using Printf
using ProgressMeter


# since we only work in the virtual space, we define level shifting 
# which allows us to shift the occupied part into the positive 
# spectrum, making it effectively invisible for solvers targeting 
# only the lowest eigenvalues.
struct LevelShiftedOperator{T, TA, TB}
    base_op::TA    # the original linear operator (A)
    subspace::TB   # the basis of the subspace to shift (V). In our case: occupied space
    μ::T           # the actual shift (μ) 
end

# define in-place multiplication for LevelShiftedOperator: Y = α*(A+μVV')*X+βY
# since many routines require this operation, we have to provide it for an LevelShiftedOperator
function LinearAlgebra.mul!(Y::AbstractMatrix, op::LevelShiftedOperator, X::AbstractMatrix, α::Number, β::Number)
    # A*X
    KX = op.base_op * X
    if β == 0
        @. Y = α * KX
    else
        @. Y = α * KX + β * Y
    end
    # μVV'*X
    overlaps = op.subspace' * X
    T = eltype(Y)
    shift_factor = T(op.μ) * T(α)
    # Y = α*(A+μVV')*X+βY
    mul!(Y, op.subspace, overlaps, shift_factor, T(1.0))
    return Y
end

# wrapper for out-of-place multiplication operator *
function Base.:*(op::LevelShiftedOperator, X::AbstractMatrix)
    Y = similar(X)
    mul!(Y, op, X, 1, 0) # Y = (A+μVV')*X
    return Y
end

# some necessary helper functions for DFTK's iterative solver to understand LevelShiftedOperator
Base.size(op::LevelShiftedOperator) = size(op.base_op)
Base.size(op::LevelShiftedOperator, i::Int) = size(op.base_op, i)
Base.eltype(op::LevelShiftedOperator) = eltype(op.base_op)
LinearAlgebra.ishermitian(op::LevelShiftedOperator) = ishermitian(op.base_op)


function main()
    pd_pbe_family = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf") # pseudopotentials

    # He atom in a box
    He = ElementPsp(:He, pd_pbe_family)
    atoms = [He]
    lattice = [[ 8.00000  0.000000  0.00000]; # units are Bohr
               [ 0.00000  7.900000  0.00000];
               [ 0.00000  0.000000  7.80000]]
    positions = [[0.500000, 0.500000, 0.500000]] # relative coordinates
    Ecut=25 # plane wave cutoff in atomic units (Hartree)

    # start with a DFT-PBE 
    model  = model_PBE(lattice, atoms, positions)
    basis  = PlaneWaveBasis(model; Ecut=Ecut, kgrid=[1, 1, 1])
    scfres_pbe = self_consistent_field(basis; tol=1e-7, maxiter=100)

    # use the PBE solution as initial guess for the HF solver
    model  = model_HF(lattice, atoms, positions)
    basis  = PlaneWaveBasis(model; Ecut=Ecut, kgrid=[1, 1, 1])
    scfres_hf = self_consistent_field(
                    basis;
                    solver=DFTK.scf_damping_solver(damping=1.0),
                    tol=1e-7, 
                    ρ=scfres_pbe.ρ, 
                    ψ=scfres_pbe.ψ, 
                    maxiter=100, 
                    diagtolalg=DFTK.AdaptiveDiagtol(; ratio_ρdiff=5e-4)
                )

    σ = 1  # non spin-polarized 
    ik = 1 # only one single k-point
    kpt = basis.kpoints[ik]
    orbitalType = eltype(scfres_hf.ψ[1]) # this is usually ComplexF64

    # we are interested in N*N_occ virtual orbitals
    N = 40

    # define the LDA exchange potential Vx(r) = - 2 * 3/4 * (3/π)^(1/3) * ρ(r)^(1/3)
    Vx = -2.0*3/4*cbrt(3/π) .* scfres_hf.ρ[:,:,:,1].^(1/3) # LDA exchange potential

    # chose initial guess (functions are defined below)
    ϕk = construct_Gaussians_according_to_potential(Vx, N, basis, kpt, orbitalType)
    #ϕk = construct_stochastic_orbitals(N, kpt, orbitalType)
    #ϕk = construct_Cholesky_of_K(scfres_hf, basis, ik, σ, N)
    #ϕk = construct_HEG_basis(N, kpt, basis, orbitalType)
    
    # Let's get the Fock exchange operator from DFTK. We do this
    # by filtering out the ExactExchange term from our basis.
    ExactExchangeTerm = only([term for term in basis.terms if term isa DFTK.TermExactExchange])
    _, K = DFTK.ene_ops(ExactExchangeTerm, basis, scfres_hf.ψ, scfres_hf.occupation) # K = exchange operator
    Kk = K[ik] # we only look at our single k-point
     
    # We also need the occupied orbitals for level shifting
    ψocc, _ = DFTK.select_occupied_orbitals(basis, scfres_hf.ψ, scfres_hf.occupation; threshold=1e-8)
    ψocck = ψocc[ik] # we only look at ik=1

    # now we set up our level shifted Fock operator
    # set level shift μ a little larger than lowest HF eigenvalue to
    # make sure the occupied part is shifted into the positive spectrum
    μ = abs(minimum(minimum.(scfres_hf.eigenvalues))) + 2.0 
    Kk_virt = LevelShiftedOperator(Kk, ψocck, μ) # shift occupied part of Fock exchange by μ

    # let's diagonalize Kk_virt in the initial guess basis
    println("diagonalize K using initial guess...")
    Kϕk = Kk_virt * ϕk
    γ = ϕk' * Kϕk
    γ = Hermitian(γ)
    eigenvalues, eigenvectors = eigen(γ)
    println(eigenvalues[1:4])

    # holy cow... this is an extremely fast and accurate iterative solver: LOBPCG 
    # can we beat it?
    println("running LOBPCG...")
    res = DFTK.lobpcg_hyper(Kk_virt, ϕk; prec=I, tol=1e-7)
    println("Eigenvalues: ", res.λ[1:4])

    # run the Davidson 
    println("running Davidson...")
    Naux = 5*N
    Σ, U = davidson(Kk_virt, Vx, ϕk, ψocck, Naux, 1e-7)
    println("Eigenvalues: ", real.(Σ[1:4]))

    # finally we construct a full basis for the exact result
    println("diagonalize K in full basis...")
    Nfull = length(kpt.G_vectors)
    println("Nfull = ", Nfull)
    φk = zeros(orbitalType, Nfull, Nfull)
    for a=1:Nfull
        φk[a,a] = 1.0
    end
    Kφk = Kk_virt * φk
    γ = φk' * Kφk
    γ = Hermitian(γ)
    eigenvalues, eigenvectors = eigen(γ)
    println("Eigenvalues: ", real.(eigenvalues[1:4]))
end


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
function davidson(
    A::LevelShiftedOperator, # linear operator
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
        ## update guess space using preconditioner 
        #t = zero(similar(R)) 
        #for i = 1:size(t,2)
        #    R_real = ifft(basis, kpt, R[:,i]) # FFT to real space
        #    C = -1.0 ./ (D_real .- Σ[i])
        #    t_real = C .* R_real # apply C
        #    t[:,i] = fft(basis, kpt, t_real) # FFT back to reciprocal space
        #end
        
        t = R # no preconditioner

        # update guess basis
        if size(V,2) <= Naux-Nlow
            V = hcat(V,t) # concatenate V and t
        else
            V = hcat(X,t) # concatenate X and t 
        end
    end
end


main()

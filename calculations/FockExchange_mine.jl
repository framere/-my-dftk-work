using Pkg
Pkg.activate(".")

using DFTK
using PseudoPotentialData
using LinearAlgebra
using StatsBase
using Printf
using ProgressMeter

# since we want to work in the virtual space only:
# we have to make sure that the virtual-occupied block is always zero
# and that the occ-occ block has very large eigenvalues (shift them above zero)
struct ProjectedShiftedOperator{TOp,T}
    base_op::TOp            # the operator
    V::T               # basis for projector
    shift::Float64     # should be a lilttle larger than the lower epsilon_hf
end

include("help_functions.jl")
include("/home/fmereto/DFTK/calculations/davdison_algo.jl")

function ProjectedShiftedOperator(base_op, V, shift)
    return ProjectedShiftedOperator{typeof(base_op), typeof(V)}(
        base_op, V, shift
    )
end
function LinearAlgebra.mul!(Y, op::ProjectedShiftedOperator, X)
    workX = similar(X)
    copy!(workX, X)
    coeffs_in = op.V' * X
    mul!(workX, op.V, coeffs_in, -1.0, 1.0) # workX -= op.V * coeffs_in
    mul!(Y, op.base_op, workX)              #     Y  = op.base_op * workX
    coeffs_out = op.V' * Y
    mul!(Y, op.V, coeffs_out, -1.0, 1.0)    #     Y -= op.V * coeffs_out
    mul!(Y, op.V, coeffs_in, op.shift, 1.0) #     Y += shift*coeffs_in
    return Y
end
function Base.:*(op::ProjectedShiftedOperator, X::AbstractMatrix)
    Y = similar(X)
    mul!(Y, op, X)
    return Y
end
Base.size(op::ProjectedShiftedOperator, args...) = size(op.base_op, args...)
Base.eltype(op::ProjectedShiftedOperator) = eltype(op.base_op)
LinearAlgebra.ishermitian(op::ProjectedShiftedOperator) = ishermitian(op.base_op)


function main()
    pd_pbe_family = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf") # pseudopotentials

    # He atom in a box
    Si = ElementPsp(:Si, pd_pbe_family)
    atoms = [Si, Si] # two Si atoms
    lattice = [[ 8.00000  0.000000  0.00000]; # units are Bohr
               [ 0.00000  7.900000  0.00000];
               [ 0.00000  0.000000  7.80000]]
    positions = [ones(3)/8, -ones(3)/8]# relative coordinates
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
    unit_cell_volume = abs(det(basis.model.lattice))
    CoulombPotential_real = irfft(basis, Complex.(ExactExchangeTerm.poisson_green_coeffs)) / sqrt(unit_cell_volume)
    v0 = CoulombPotential_real[1,1,1] # Coulomb potential at r=0

    # diagonal K(r,r) = ρ(r) * v(0)
    D_real = scfres_hf.ρ[:,:,:,σ] .* v0

    _, K = DFTK.ene_ops(ExactExchangeTerm, basis, scfres_hf.ψ, scfres_hf.occupation) # K = exchange operator
    Kk = K[ik] # we only look at our single k-point
     
    # We also need the occupied orbitals for level shifting
    ψocc, _ = DFTK.select_occupied_orbitals(basis, scfres_hf.ψ, scfres_hf.occupation; threshold=1e-8)
    ψocck = ψocc[ik] # we only look at ik=1

    # now we set up our level shifted Fock operator
    # set level shift μ a little larger than lowest HF eigenvalue to
    # make sure the occupied part is shifted into the positive spectrum
    μ = abs(minimum(minimum.(scfres_hf.eigenvalues))) + 2.0 
    Kk_virt = ProjectedShiftedOperator(Kk, ψocck, μ) # shift occupied part of Fock exchange by μ

    # let's diagonalize Kk_virt in the initial guess basis
    println("diagonalize K using initial guess...")
    Kϕk = Kk_virt * ϕk
    γ = ϕk' * Kϕk
    γ = Hermitian(γ)
    eigenvalues_γ, eigenvectors_γ = eigen(γ)
    println(eigenvalues_γ[1:4])


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
    @time eigenvalues, eigenvectors = eigen(γ)
    println("Eigenvalues: ", real.(eigenvalues[1:4]))

    # holy cow... this is an extremely fast and accurate iterative solver: LOBPCG 
    # can we beat it?
    println("running LOBPCG...")
    @time res = DFTK.lobpcg_hyper(Kk_virt, ϕk; prec=I, tol=1e-6)
    display("text/plain", (res.λ[1:N] - eigenvalues[1:N])')

    # run the Davidson 
    println("running Davidson...")
    Naux = 10*N
    l = 40                     # number of desired eigenpairs
    thresh = 1e-5
    max_iter = 200
    sorted_indices = sortperm(abs.(eigenvalues_γ), rev=true)  # or sortperm(eigenvalues_γ) for ascending
    eigenvalues_γ = eigenvalues_γ[sorted_indices]

    # Use first Nlow eigenvectors as initial guess
    V_full = ϕk * eigenvectors_γ[:, 1:N]
    V = V_full[:, 1:10]  # pass only l initial guess vectors

    # Pass all sorted indices to davidson
    all_idxs = sorted_indices

    # @time Σ, U = davidson(Kk_virt, D_real, V, V_full, Naux, l, thresh, max_iter, all_idxs)
    # idx = sortperm(Σ)
    # Σ = Σ[idx]

    # display("text/plain", (Σ[1:N] - eigenvalues[1:N])')

    @time Σ_ref, U_ref = davidson_tobias(Kk_virt, D_real, V_full, ψocck, Naux, thresh)

    idx_ref = sortperm(real.(Σ_ref))
    Σ_ref = Σ_ref[idx_ref]
    display("text/plain", (real.(Σ_ref[1:N]) - eigenvalues[1:N])')
end

main()

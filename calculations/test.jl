using Pkg
Pkg.activate(".") # activate the environment

using DFTK # this will now use the cloned DFTK instead of the official package
using PseudoPotentialData

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
    println(scfres_hf.energies)
    println("should be -2.803888535")
end

main()

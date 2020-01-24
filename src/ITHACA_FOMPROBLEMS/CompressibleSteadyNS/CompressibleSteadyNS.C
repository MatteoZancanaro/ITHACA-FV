/*---------------------------------------------------------------------------*\
     ██╗████████╗██╗  ██╗ █████╗  ██████╗ █████╗       ███████╗██╗   ██╗
     ██║╚══██╔══╝██║  ██║██╔══██╗██╔════╝██╔══██╗      ██╔════╝██║   ██║
     ██║   ██║   ███████║███████║██║     ███████║█████╗█████╗  ██║   ██║
     ██║   ██║   ██╔══██║██╔══██║██║     ██╔══██║╚════╝██╔══╝  ╚██╗ ██╔╝
     ██║   ██║   ██║  ██║██║  ██║╚██████╗██║  ██║      ██║      ╚████╔╝
     ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝      ╚═╝       ╚═══╝

 * In real Time Highly Advanced Computational Applications for Finite Volumes
 * Copyright (C) 2017 by the ITHACA-FV authors
-------------------------------------------------------------------------------

  License
  This file is part of ITHACA-FV

  ITHACA-FV is free software: you can redistribute it and/or modify
  it under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  ITHACA-FV is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with ITHACA-FV. If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/


/// \file
/// Source file of the steadyNS class.

#include "CompressibleSteadyNS.H"
#include "viscosityModel.H"

// * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * * * //

// Constructor
CompressibleSteadyNS::CompressibleSteadyNS() {}
CompressibleSteadyNS::CompressibleSteadyNS(int argc, char* argv[])
{
    //#include "postProcess.H"
    _args = autoPtr<argList>
            (
                new argList(argc, argv)
            );

    if (!_args->checkRootCase())
    {
        Foam::FatalError.exit();
    }

    argList& args = _args();
#include "createTime.H"
#include "createMesh.H"
    _simple = autoPtr<simpleControl>
              (
                  new simpleControl
                  (
                      mesh
                  )
              );
    simpleControl& simple = _simple();
#include "createFields.H"
    //#include "createFvOptions.H"
#include "initContinuityErrs.H"
    supex = ITHACAutilities::check_sup();
    turbulence->validate();
    ITHACAdict = new IOdictionary
    (
        IOobject
        (
            "ITHACAdict",
            runTime.system(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );
    para = new ITHACAparameters;
    offline = ITHACAutilities::check_off();
    podex = ITHACAutilities::check_pod();
}


// * * * * * * * * * * * * * * Full Order Methods * * * * * * * * * * * * * * //

// Method to perform a truthSolve

void CompressibleSteadyNS::truthSolve()
{
    Time& runTime = _runTime();
    volScalarField& E = pThermo().he();
    //volScalarField& E = _E();
    volVectorField& U = _U();
    simpleControl& simple = _simple();
    volScalarField& p = pThermo().p();
    volScalarField _nut(turbulence->nut());
    //Info << thermo.mu() << endl;
    //Info << thermo.nu() << endl;
#include "NLsolve.H"
    ITHACAstream::exportSolution(U, name(counter), "./ITHACAoutput/Offline/");
    ITHACAstream::exportSolution(p, name(counter), "./ITHACAoutput/Offline/");
    ITHACAstream::exportSolution(E, name(counter), "./ITHACAoutput/Offline/");
    ITHACAstream::exportSolution(_nut, name(counter), "./ITHACAoutput/Offline/");
    Ufield.append(U);
    Pfield.append(p);
    Efield.append(E);
    nutFields.append(_nut);
    counter++;
}

void CompressibleSteadyNS::changeViscosity(double mu_new)
{
    const volScalarField& mu =  pThermo().mu();
    volScalarField& mu_field = const_cast<volScalarField&>(mu);
    this->assignIF(mu_field, mu_new);

    for (int i = 0; i < mu_field.boundaryFieldRef().size(); i++)
    {
        this->assignBC(mu_field, i, mu_new);
    }
}

fvVectorMatrix CompressibleSteadyNS::getNLTerm(volVectorField& U)
{
    surfaceScalarField& phi = _phi();
    fvVectorMatrix NLTerm = fvm::div(phi, U);
    return NLTerm;
}

fvVectorMatrix CompressibleSteadyNS::getViscTerm(volVectorField& U)
{
    volScalarField& rho = _rho();
    volScalarField nuEff = turbulence->nuEff();
    //fvVectorMatrix viscTerm = turbulence->divDevRhoReff(U);
    fvVectorMatrix viscTerm = - fvc::div((rho * nuEff) * dev2(T(fvc::grad(
            U)))) - fvm::laplacian(rho * nuEff, U);
    return viscTerm;
}

volVectorField CompressibleSteadyNS::getGradP(volScalarField& p)
{
    volVectorField gradP = fvc::grad(p);
    return gradP;
}


fvVectorMatrix CompressibleSteadyNS::getUmatrix(volVectorField&
        U)//, Vector<double>& uresidual_v)
{
    volScalarField& rho = _rho();
    fv::options& fvOptions = _fvOptions();
    Ueqn_global.reset(new fvVectorMatrix(getNLTerm(U)
                                         + getViscTerm(U)
                                         ==
                                         fvOptions(rho, U)
                                        ));
    Ueqn_global().relax();
    fvOptions.constrain(Ueqn_global());
    //Ueqn_global.reset(new fvVectorMatrix(Ueqn_global()
    //                                  ==
    //                                  -getGradP(p)
    //                                 ));
    return Ueqn_global();
}

fvScalarMatrix CompressibleSteadyNS::getFluxTerm()
{
    volScalarField& he = pThermo().he();
    surfaceScalarField& phi = _phi();
    fvScalarMatrix fluxTerm = fvm::div(phi, he);
    return fluxTerm;
}

volScalarField CompressibleSteadyNS::getKinEnTerm(volVectorField& U,
        volScalarField& p)
{
    surfaceScalarField& phi = _phi();
    volScalarField& rho = _rho();
    volScalarField kinEn = fvc::div(phi, volScalarField("Ekp",
                                    0.5 * magSqr(U) + p / rho));
    return kinEn;
}

fvScalarMatrix CompressibleSteadyNS::getDiffTerm()
{
    volScalarField& he = pThermo().he();
    fvScalarMatrix diffTerm = fvm::laplacian(turbulence->alphaEff(), he);
    return diffTerm;
}

fvScalarMatrix CompressibleSteadyNS::getEmatrix(volVectorField& U,
        volScalarField& p)//, scalar& eresidual)
{
    // fluidThermo& thermo = pThermo();
    // volScalarField& he = thermo.he();
    volScalarField& he = pThermo().he();
    volScalarField& rho = _rho();
    fv::options& fvOptions = _fvOptions();
    Eeqn_global.reset(new fvScalarMatrix(
                          getFluxTerm() + getKinEnTerm(U, p) - getDiffTerm()
                          ==
                          fvOptions(rho, he)
                      ));
    Eeqn_global().relax();
    fvOptions.constrain(Eeqn_global());
    return Eeqn_global();
}

surfaceScalarField CompressibleSteadyNS::getPhiHbyA(fvVectorMatrix& Ueqn,
        volVectorField& U, volScalarField& p)
{
    volScalarField& rho = _rho();
    volScalarField rAU(1.0 /
                       Ueqn.A()); // Inverse of the diagonal part of the U equation matrix
    HbyA.reset(new volVectorField(constrainHbyA(rAU * Ueqn.H(), U,
                                  p))); // H is the extra diagonal part summed to the r.h.s. of the U equation
    phiHbyA.reset(new surfaceScalarField("phiHbyA",
                                         fvc::interpolate(rho)*fvc::flux(HbyA)));
    return phiHbyA;
}

volScalarField CompressibleSteadyNS::getDivPhiHbyA(fvVectorMatrix& Ueqn,
        volVectorField& U, volScalarField& p)
{
    volScalarField divPhiHbyA = fvc::div(getPhiHbyA(Ueqn, U, p));
    return divPhiHbyA;
}

surfaceScalarField CompressibleSteadyNS::getRhorAUf(fvVectorMatrix& Ueqn)
{
    volScalarField& rho = _rho();
    volScalarField rAU(1.0 /
                       Ueqn.A()); // Inverse of the diagonal part of the U equation matrix
    rhorAUf.reset(new surfaceScalarField("rhorAUf", fvc::interpolate(rho * rAU)));
    return rhorAUf;
}

fvScalarMatrix CompressibleSteadyNS::getPoissonTerm(fvVectorMatrix& Ueqn, volScalarField& p)
{
    fvScalarMatrix poissonTerm = fvm::laplacian(getRhorAUf(Ueqn), p);
    return poissonTerm;
}

fvScalarMatrix CompressibleSteadyNS::getPmatrix(fvVectorMatrix& Ueqn, volVectorField& U, volScalarField& p)
{
    volScalarField& rho = _rho();
    fv::options& fvOptions = _fvOptions();
    volScalarField& psi = _psi();
    pressureControl& pressureControl = _pressureControl();
    Peqn_global.reset(new fvScalarMatrix(
                          getDivPhiHbyA(Ueqn, U, p)
                          - getPoissonTerm(Ueqn, p)
                          ==
                          fvOptions(psi, p, rho.name())
                      ));
    Peqn_global().setReference
    (
        pressureControl.refCell(),
        pressureControl.refValue()
    );
    return Peqn_global();
}

void CompressibleSteadyNS::restart()
{
    _runTime().objectRegistry::clear();
    _mesh().objectRegistry::clear();
    // _mesh.clear();
    // _runTime.clear();
    _simple.clear();
    pThermo.clear();
    _p.clear();
    _rho.clear();
    _E.clear();
    _U.clear();
    _phi.clear();
    turbulence.clear();
    _initialMass.clear();
    _pressureControl.clear();
    _psi.clear();
    _fvOptions.clear();
    argList& args = _args();
    Info << "ReCreate time\n" << Foam::endl;
    // _runTime = autoPtr<Foam::Time>( new Foam::Time( Foam::Time::controlDictName,
    //                                 args ) );
    // std::cerr << "File: CompressibleSteadyNS.C, Line: 281" << std::endl;
    Time& runTime = _runTime();
    // _mesh = autoPtr<fvMesh>
    //         (
    //             new fvMesh
    //             (
    //                 IOobject
    //                 (
    //                     fvMesh::defaultRegion,
    //                     runTime.timeName(),
    //                     runTime,
    //                     IOobject::MUST_READ
    //                 )
    //             )
    //         );
    Foam::fvMesh& mesh = _mesh();
    _simple = autoPtr<simpleControl>
              (
                  new simpleControl
                  (
                      mesh
                  )
              );
    simpleControl& simple = _simple();
    pThermo = autoPtr<fluidThermo>
              (
                  fluidThermo::New(mesh)
              );
    fluidThermo& thermo = pThermo();
    thermo.validate(_args().executable(), "h", "e");
    _p = autoPtr<volScalarField>
         (
             new volScalarField(thermo.p())
         );
    volScalarField& p = thermo.p();
    _rho = autoPtr<volScalarField>
           (
               new volScalarField
               (
                   IOobject
                   (
                       "rho",
                       runTime.timeName(),
                       mesh,
                       IOobject::READ_IF_PRESENT,
                       IOobject::AUTO_WRITE
                   ),
                   thermo.rho()
               )
           );
    volScalarField& rho = _rho();
    _E = autoPtr<volScalarField>
         (
             new volScalarField(thermo.he())
         );
    Info << "ReReading field U\n" << endl;
    _U = autoPtr<volVectorField>
         (
             new volVectorField
             (
                 IOobject
                 (
                     "U",
                     runTime.timeName(),
                     mesh,
                     IOobject::MUST_READ,
                     IOobject::AUTO_WRITE
                 ),
                 mesh
             )
         );
    volVectorField& U = _U();
    Info << "ReReading/calculating face flux field phi\n" << endl;
    _phi = autoPtr<surfaceScalarField>
           (
               new surfaceScalarField
               (
                   IOobject
                   (
                       "phi",
                       runTime.timeName(),
                       mesh,
                       IOobject::READ_IF_PRESENT,
                       IOobject::AUTO_WRITE
                   ),
                   linearInterpolate(rho * U) & mesh.Sf()
               )
           );
    surfaceScalarField& phi = _phi();
    _pressureControl = autoPtr<pressureControl>
                       (
                           new pressureControl(p, rho, _simple().dict())
                       );
    mesh.setFluxRequired(p.name());
    Info << "ReCreating turbulence model\n" << endl;
    turbulence = autoPtr<compressible::turbulenceModel>
                 (
                     compressible::turbulenceModel::New
                     (
                         rho,
                         U,
                         phi,
                         thermo
                     )
                 );
    _initialMass = autoPtr<dimensionedScalar>
                   (
                       new dimensionedScalar(fvc::domainIntegrate(rho))
                   );
    _psi = autoPtr<volScalarField>
           (
               new volScalarField(thermo.psi())
           );

        _fvOptions = autoPtr<fv::options>(new fv::options(mesh));
#include "initContinuityErrs.H"
    //turbulence->validate();
}

// fvScalarMatrix CompressibleSteadyNS::getPmatrix(volVectorField& U,
//         volScalarField& p, scalar& presidual, fvVectorMatrix& Ueqn)
// {
//     surfaceScalarField& phi = _phi();
//     volScalarField& rho = _rho();
//     fv::options& fvOptions = _fvOptions();
//     simpleControl& simple = _simple();
//     fluidThermo& thermo = pThermo();
//     volScalarField& psi = _psi();
//     pressureControl& pressureControl = _pressureControl();
//     Time& runTime = _runTime();
//     fvMesh& mesh = _mesh();
//     dimensionedScalar& initialMass = _initialMass();
//     bool closedVolume = false;

//     // Update the pressure BCs to ensure flux consistency
//     constrainPressure(p, rho, U, getPhiHbyA(Ueqn, U, p), getRhorAUf(Ueqn));

//     closedVolume = adjustPhi(phiHbyA(), U, p);
//     while (simple.correctNonOrthogonal())
//     {
//         // Peqn_global.reset(new fvScalarMatrix(
//         //                       fvc::div(phiHbyA)
//         //                       - fvm::laplacian(rhorAUf, p)
//         //                       ==
//         //                       fvOptions(psi, p, rho.name())
//         //                   ));
//         Peqn_global.reset(new fvScalarMatrix(
//                               getDivPhiHbyA()
//                               - getPoissonTerm(p)
//                               ==
//                               fvOptions(psi, p, rho.name())
//                           ));
//         Peqn_global().setReference
//         (
//             pressureControl.refCell(),
//             pressureControl.refValue()
//         );
//         presidual = Peqn_global().solve().initialResidual();

//         if (simple.finalNonOrthogonalIter())
//         {
//             phi = phiHbyA() + Peqn_global().flux();
//         }
//     }

// #include "incompressible/continuityErrs.H"
//     // Explicitly relax pressure for momentum corrector
//     p.relax();
//     U = HbyA() - (1.0 / Ueqn.A()) * fvc::grad(p);//rAU * fvc::grad(p);
//     U.correctBoundaryConditions();
//     fvOptions.correct(U);
//     bool pLimited = pressureControl.limit(p);

//     // For closed-volume cases adjust the pressure and density levels
//     // to obey overall mass continuity
//     if (closedVolume)
//     {
//         p += (initialMass - fvc::domainIntegrate(psi * p))
//              / fvc::domainIntegrate(psi);
//     }

//     if (pLimited || closedVolume)
//     {
//         p.correctBoundaryConditions();
//     }

//     rho = thermo.rho(); // Here rho is calculated as p*psi = p/(R*T)
//     rho.relax();

//     return Peqn_global();
// }

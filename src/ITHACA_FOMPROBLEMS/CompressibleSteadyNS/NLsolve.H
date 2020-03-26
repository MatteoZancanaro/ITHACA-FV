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
fv::options& fvOptions = _fvOptions();
surfaceScalarField& phi = _phi();
volScalarField& rho = _rho();
fluidThermo& thermo = pThermo();
volScalarField& psi = _psi();
pressureControl& pressureControl = _pressureControl();
fvMesh& mesh = _mesh();
dimensionedScalar& initialMass = _initialMass();
bool closedVolume = false;

scalar residual = 1;
scalar eresidual = 1;
Vector<double> uresidual_v(0, 0, 0);
scalar presidual = 1;

scalar csolve = 0;
scalar saver = 0;
scalar folderN = 0;


// Variable that can be changed
std::ofstream res_os;
res_os.open("./ITHACAoutput/Offline/residuals", std::ios_base::app);

tolerance = ITHACAdict->lookupOrDefault<scalar>("tolerance", 1e-5);
maxIter = ITHACAdict->lookupOrDefault<scalar>("maxIter", 2000);

turbulence->validate();

#if OFVER == 6

while (simple.loop(runTime) && residual > tolerance && csolve < maxIter )
#else
while (simple.loop() && residual > tolerance && csolve < maxIter )
#endif
{
    Info << "Time = " << runTime.timeName() << nl << endl;
    //Momentum equation phase
    getUmatrix(U);

    if (simple.momentumPredictor())
    {
        uresidual_v = solve(Ueqn_global() == - getGradP(p)).initialResidual(); //Working
        // dimensionedVector Zer("Z", dimForce, vector(0,0,0));
        // dimensionedScalar Vol("V", dimVolume, 1.0);
        // dimensionedVector zero = Zer/Vol;
        // uresidual_v = solve(Ueqn_global() == zero).initialResidual(); //Not working
        //uresidual_v = Ueqn_global().solve().initialResidual(); //Not working
        //uresidual_v = getUmatrix(U, p).solve().initialResidual(); //Not working
        fvOptions.correct(U);
    }

    //Energy equation phase
    getEmatrix(U, p);
    eresidual = Eeqn_global().solve().initialResidual();
    //Info << E[100] << endl;
    //Info << E[1000] << endl;
    //Info << E[5000] << endl;
    fvOptions.correct(thermo.he());
    thermo.correct(); // Here are calculated both temperature and density based on P,U and he.
    // Pressure equation phase
    constrainPressure(p, rho, U, getPhiHbyA(Ueqn_global(), U, p),
                      getRhorAUf(Ueqn_global()));// Update the pressure BCs to ensure flux consistency
    //closedVolume = adjustPhi(phiHbyA(), U, p);
    //closedVolume = adjustPhi(getPhiHbyA(Ueqn_global(), U, p), U, p);
    surfaceScalarField phiHbyACalculated = getPhiHbyA(Ueqn_global(), U, p);
    closedVolume = adjustPhi(phiHbyACalculated, U, p);


    while (simple.correctNonOrthogonal())
    {
        getPmatrix(Ueqn_global(), U, p);
        presidual = Peqn_global().solve().initialResidual();

        if (simple.finalNonOrthogonalIter())
        {
            //phi = phiHbyA() + Peqn_global().flux();
            phi = getPhiHbyA(Ueqn_global(), U, p) + Peqn_global().flux();
        }
    }

#include "incompressible/continuityErrs.H"
    p.relax();// Explicitly relax pressure for momentum corrector
    U = HbyA() - (1.0 / Ueqn_global().A()) * getGradP(p);//rAU * fvc::grad(p);
    U.correctBoundaryConditions();
    fvOptions.correct(U);
    bool pLimited = pressureControl.limit(p);

    // For closed-volume cases adjust the pressure and density levels to obey overall mass continuity
    if (closedVolume)
    {
        p += (initialMass - fvc::domainIntegrate(psi * p))
             / fvc::domainIntegrate(psi);
    }

    if (pLimited || closedVolume)
    {
        p.correctBoundaryConditions();
    }

    rho = thermo.rho(); // Here rho is calculated as p*psi = p/(R*T)
    rho.relax();
    auto uresidual = max(max(uresidual_v[0], uresidual_v[1]), uresidual_v[2]);
    residual = max(max(presidual, uresidual), eresidual);
    Info << "\nResidual: " << residual << endl << endl;
    turbulence->correct();
    csolve = csolve + 1;
    Info << "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
         << "  ClockTime = " << runTime.elapsedClockTime() << " s"
         << nl << endl;

    saver++;
    //std::cout << saver << std::endl;
    if (saver==20)
    {
      saver = 0;
      folderN++;
      ITHACAstream::exportSolution(U, name(folderN), "./ITHACAoutput/Offline/");
      ITHACAstream::exportSolution(p, name(folderN), "./ITHACAoutput/Offline/");
      ITHACAstream::exportSolution(E, name(folderN), "./ITHACAoutput/Offline/");
      ITHACAstream::exportSolution(_nut, name(folderN), "./ITHACAoutput/Offline/");
      Ufield.append(U);
      Pfield.append(p);
      Efield.append(E);
      nutFields.append(_nut);
    }
}

ITHACAstream::exportSolution(U, name(folderN+1), "./ITHACAoutput/Offline/");
ITHACAstream::exportSolution(p, name(folderN+1), "./ITHACAoutput/Offline/");
ITHACAstream::exportSolution(E, name(folderN+1), "./ITHACAoutput/Offline/");
ITHACAstream::exportSolution(_nut, name(folderN+1), "./ITHACAoutput/Offline/");
Ufield.append(U);
Pfield.append(p);
Efield.append(E);
nutFields.append(_nut);

res_os << residual << std::endl;
res_os.close();
runTime.setTime(runTime.startTime(), 0);









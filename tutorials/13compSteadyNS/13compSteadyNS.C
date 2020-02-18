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
Description
    Example of steady NS Reduction Problem
SourceFiles
    03steadyNS.C
\*---------------------------------------------------------------------------*/

#include "CompressibleSteadyNS.H"
#include "ReducedCompressibleSteadyNS.H"
#include "ITHACAstream.H"
#include "ITHACAPOD.H"
#include "forces.H"
#include "IOmanip.H"


class tutorial13 : public CompressibleSteadyNS
{
public:
    /// Constructor
    explicit tutorial13(int argc, char* argv[])
        :
        CompressibleSteadyNS(argc, argv),
        U(_U()),
        p(_p()),
        E(_E())
    {}

    /// Velocity field
    volVectorField& U;
    /// Pressure field
    volScalarField& p;
    /// Energy field
    volScalarField& E;

    /// Perform an Offline solve
    void offlineSolve()
    {
        //Vector<double> inl(0, 0, 0);
        //List<scalar> mu_now(1);

        // if the offline solution is already performed read the fields
        if (offline)
        {
            ITHACAstream::read_fields(Ufield, U, "./ITHACAoutput/Offline/");
            ITHACAstream::read_fields(Pfield, p, "./ITHACAoutput/Offline/");
            ITHACAstream::read_fields(Efield, E, "./ITHACAoutput/Offline/");
            mu_samples = ITHACAstream::readMatrix("./parsOff_mat.txt");
        }
        // else perform offline stage
        else
        {
            Vector<double> Uinl(250, 0, 0);

            for (label i = 0; i < mu.rows(); i++)
            {
                std::cout << "Current mu = " << mu(i, 0) << std::endl;
                //mu_now[0] = mu(i, 0);
                //changeViscosity(mu_now[0]);
                changeViscosity(mu(i, 0));
                assignIF(U, Uinl);
                //truthSolve(mu_now);
                truthSolve();
            }
        }
    }

};

int main(int argc, char* argv[])
{
    // Construct the tutorial object
    tutorial13 example(argc, argv);

    //Eigen::MatrixXd parOff;
    std::ifstream exFileOff("./parsOff_mat.txt");
    if (exFileOff)
    {
        example.mu  = ITHACAstream::readMatrix("./parsOff_mat.txt");
    }

    else
    {
        //example.mu  = ITHACAutilities::rand(20, 1, 1.00e-05, 1.00e-2);
        example.mu  = Eigen::VectorXd::LinSpaced(50, 1.00e-05, 1.00e-02);
        ITHACAstream::exportMatrix(example.mu , "parsOff", "eigen", "./");
    }

    Eigen::MatrixXd parOn;
    std::ifstream exFileOn("./parsOn_mat.txt");
    if (exFileOn)
    {
        parOn = ITHACAstream::readMatrix("./parsOn_mat.txt");
    }

    else
    {
        parOn = ITHACAutilities::rand(20, 1, 1.00e-05, 1.00e-02);
        ITHACAstream::exportMatrix(parOn, "parsOn", "eigen", "./");
    }

    // Read some parameters from file
    ITHACAparameters para;
    int NmodesUout = para.ITHACAdict->lookupOrDefault<int>("NmodesUout", 15);
    int NmodesPout = para.ITHACAdict->lookupOrDefault<int>("NmodesPout", 15);
    int NmodesEout = para.ITHACAdict->lookupOrDefault<int>("NmodesEout", 15);
    int NmodesUproj = para.ITHACAdict->lookupOrDefault<int>("NmodesUproj", 10);
    int NmodesPproj = para.ITHACAdict->lookupOrDefault<int>("NmodesPproj", 10);
    int NmodesEproj = para.ITHACAdict->lookupOrDefault<int>("NmodesEproj", 10);
    //Set the inlet boundaries patch 0 directions x and y
    example.inletIndex.resize(1, 2);
    example.inletIndex(0, 0) = 1;
    example.inletIndex(0, 1) = 0;
    //Perform the offline solve
    example.offlineSolve();
    //Read the lift field
    ITHACAstream::read_fields(example.liftfield, example.U, "./lift/");
    ITHACAutilities::normalizeFields(example.liftfield);
    // Homogenize the snapshots
    example.computeLift(example.Ufield, example.liftfield, example.Uomfield);
    // Perform POD on velocity and pressure and store the first 10 modes
    ITHACAPOD::getModes(example.Uomfield, example.Umodes, example.podex, 0, 0,
                        NmodesUout);
    ITHACAPOD::getModes(example.Pfield, example.Pmodes, example.podex, 0, 0,
                        NmodesPout);
    ITHACAPOD::getModes(example.Efield, example.Emodes, example.podex, 0, 0,
                        NmodesEout);
    // Create the reduced object
    ReducedCompressibleSteadyNS reduced(example);
    PtrList<volVectorField> uFull;
    ITHACAstream::read_fields(uFull, example.U, "./ITHACAoutput/Offline/");
    Eigen::MatrixXd projU = ITHACAutilities::getCoeffsFrobenius(uFull, reduced.ULmodes, 10);
    //std::cout << projU << std::endl;
    PtrList<volVectorField> projectedU = ITHACAutilities::reconstruct_from_coeff(reduced.ULmodes, projU, 10);
    ITHACAstream::exportFields(projectedU, "./ITHACAoutput/Offline/", "projU");
    // Reads inlet volocities boundary conditions.
    word vel_file(para.ITHACAdict->lookup("online_velocities"));
    Eigen::MatrixXd vel = ITHACAstream::readMatrix(vel_file);

    //Perform the online solutions
    for (label k = 0; k < parOn.rows(); k++)
    {
        scalar mu_now = parOn(k, 0);
        example.changeViscosity(mu_now);
        reduced.setOnlineVelocity(vel);
        reduced.solveOnlineCompressible(mu_now, NmodesUproj, NmodesPproj, NmodesEproj);
    }
    exit(0);
}
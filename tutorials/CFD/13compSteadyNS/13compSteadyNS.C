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
    void offlineSolve(word folder = "./ITHACAoutput/Offline/")
    {
        //Vector<double> inl(0, 0, 0);
        //List<scalar> mu_now(1);

        // if the offline solution is already performed read the fields
        if (offline)
        {
            ITHACAstream::readMiddleFields(Ufield, U, folder);
            ITHACAstream::readMiddleFields(Pfield, p, folder);
            ITHACAstream::readMiddleFields(Efield, E, folder);
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
                truthSolve(folder);
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
    ITHACAparameters* para = ITHACAparameters::getInstance(example._mesh(),
                             example._runTime());
    int NmodesUout = para->ITHACAdict->lookupOrDefault<int>("NmodesUout", 15);
    int NmodesPout = para->ITHACAdict->lookupOrDefault<int>("NmodesPout", 15);
    int NmodesEout = para->ITHACAdict->lookupOrDefault<int>("NmodesEout", 15);
    int NmodesUproj = para->ITHACAdict->lookupOrDefault<int>("NmodesUproj", 10);
    int NmodesPproj = para->ITHACAdict->lookupOrDefault<int>("NmodesPproj", 10);
    int NmodesEproj = para->ITHACAdict->lookupOrDefault<int>("NmodesEproj", 10);
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
    ITHACAPOD::getModes(example.Uomfield, example.Umodes, example._U().name(), example.podex, 0, 0,
                        NmodesUout);
    ITHACAPOD::getModes(example.Pfield, example.Pmodes, example._p().name(), example.podex, 0, 0,
                        NmodesPout);
    ITHACAPOD::getModes(example.Efield, example.Emodes, example._E().name(), example.podex, 0, 0,
                        NmodesEout);
    // Create the reduced object
    ReducedCompressibleSteadyNS reduced(example);
    PtrList<volVectorField> uFull;
    ITHACAstream::read_fields(uFull, example.U, "./ITHACAoutput/Offline/");
    Eigen::MatrixXd projU = ITHACAutilities::getCoeffs(uFull, reduced.ULmodes, 10, false);
    //std::cout << projU << std::endl;
    PtrList<volVectorField> projectedU = ITHACAutilities::reconstructFromCoeff(reduced.ULmodes, projU, 10);
    ITHACAstream::exportFields(projectedU, "./ITHACAoutput/Offline/", "projU");
    // Reads inlet volocities boundary conditions.
    word vel_file(para->ITHACAdict->lookup("online_velocities"));
    Eigen::MatrixXd vel = ITHACAstream::readMatrix(vel_file);

    //Perform the online solutions
    for (label k = 0; k < parOn.rows(); k++)
    {
        scalar mu_now = parOn(k, 0);
        example.changeViscosity(mu_now);
        reduced.setOnlineVelocity(vel);
        reduced.projectReducedOperators(NmodesUproj, NmodesPproj, NmodesEproj);
        reduced.solveOnlineCompressible(mu_now, NmodesUproj, NmodesPproj, NmodesEproj);
    }

    // Error analysis
    tutorial13 checkOff(argc, argv);

    if (!ITHACAutilities::check_folder("./ITHACAoutput/checkOff"))
    {
        std::cerr << "debug point 1" << std::endl;
        checkOff.restart();
        std::cerr << "debug point 2" << std::endl;
        // ITHACAparameters* para = ITHACAparameters::getInstance(checkOff._mesh(),
        //                          checkOff._runTime());
        checkOff.offline = false;
        checkOff.mu = parOn;
        checkOff.offlineSolve("./ITHACAoutput/checkOff/");
        checkOff.offline = true;
    }
    std::cerr << "debug point 1" << std::endl;
    PtrList<volVectorField> Ufull;
    PtrList<volScalarField> Pfull;
    PtrList<volScalarField> Efull;
    PtrList<volVectorField> Ured;
    PtrList<volScalarField> Pred;
    PtrList<volScalarField> Ered;
    // volVectorField U("Uaux", checkOff._U());
    // volScalarField p("Paux", checkOff._p());
    // volScalarField e("Eaux", checkOff._E());
    ITHACAstream::read_fields(Ufull, checkOff._U(),
                                      "./ITHACAoutput/checkOff/");
    ITHACAstream::read_fields(Pfull, checkOff._p(),
                                      "./ITHACAoutput/checkOff/");
    ITHACAstream::read_fields(Efull, checkOff._E(),
                                      "./ITHACAoutput/checkOff/");
    ITHACAstream::read_fields(Ured, checkOff._U(), "./ITHACAoutput/Online/");
    ITHACAstream::read_fields(Pred, checkOff._p(), "./ITHACAoutput/Online/");
    ITHACAstream::read_fields(Ered, checkOff._E(), "./ITHACAoutput/Online/");
    Eigen::MatrixXd relErrorU(Ufull.size(), 1);
    Eigen::MatrixXd relErrorP(Pfull.size(), 1);
    Eigen::MatrixXd relErrorE(Efull.size(), 1);
    dimensionedVector U_fs("U_fs", dimVelocity, vector(1, 0, 0));

    for (label k = 0; k < Ufull.size(); k++)
    {
        volVectorField errorU = Ufull[k] - Ured[k];
        volVectorField devU = Ufull[k] - U_fs;
        volScalarField errorP = Pfull[k] - Pred[k];
        volScalarField errorE = Efull[k] - Ered[k];
        relErrorU(k, 0) = ITHACAutilities::frobNorm(errorU) /
                          ITHACAutilities::frobNorm(devU);
        relErrorP(k, 0) = ITHACAutilities::frobNorm(errorP) /
                          ITHACAutilities::frobNorm(Pfull[k]);
        relErrorE(k, 0) = ITHACAutilities::frobNorm(errorE) /
                          ITHACAutilities::frobNorm(Efull[k]);
    }

    ITHACAstream::exportMatrix(relErrorU,
                               "errorU_" + name(NmodesUproj) + "_" + name(NmodesPproj) + "_" + name(NmodesEproj), "python", ".");
    ITHACAstream::exportMatrix(relErrorP,
                               "errorP_" + name(NmodesUproj) + "_" + name(NmodesPproj) + "_" + name(NmodesEproj), "python", ".");
    ITHACAstream::exportMatrix(relErrorE,
                               "errorE_" + name(NmodesUproj) + "_" + name(NmodesPproj) + "_" + name(NmodesEproj), "python", ".");
    exit(0);
}
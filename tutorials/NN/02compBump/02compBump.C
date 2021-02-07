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
#include "RBFMotionSolver.H"
#include "ITHACAstream.H"
#include "ITHACAPOD.H"
#include "forces.H"
#include "IOmanip.H"


class tutorial02 : public CompressibleSteadyNS
{
public:
    /// Constructor
    explicit tutorial02(int argc, char* argv[])
        :
        CompressibleSteadyNS(argc, argv)
    {
    	dyndict = new IOdictionary
            (
                IOobject
                (
                    "dynamicMeshDictRBF",
                    "./constant",
                    _mesh(),
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE
                )
            );
        ITHACAutilities::getPointsFromPatch(_mesh(), 0, top0, top0_ind);
        ITHACAutilities::getPointsFromPatch(_mesh(), 1, bot0, bot0_ind);
        std::cout << _mesh().points().size() << std::endl;
        ms = new RBFMotionSolver(_mesh(), *dyndict);
        vectorField motion(ms->movingPoints().size(), vector::zero);
        movingIDs = ms->movingIDs();
        x0 = ms->movingPoints();
        curX = x0;
        point0 = ms->curPoints();
    }

    List<vector> top0;
    List<vector> bot0;
    labelList top0_ind;
    labelList bot0_ind;
    IOdictionary* dyndict;
    RBFMotionSolver* ms;
    labelList movingIDs;
    List<vector> x0;
    List<vector> curX;
    vectorField point0;
    vectorField point;

	double f1(double chord, double x)
		{
		    double res = chord * (std::pow((x)/chord,0.5)*(1-(x)/chord))/(std::exp(15*(x)/chord));
		    return res;
		}

	List<vector> moveBasis(const List<vector>& originalPoints, double par)
		{
		        List<vector> movedPoints(originalPoints);
		        for(int i = 0; i<originalPoints.size(); i++)
		        {
		            movedPoints[i][2]+= par*f1(1,movedPoints[i][0]);
		        }		

		        return movedPoints;
		}

    void updateMesh(double parTop, double parBot)
        {
            _mesh().movePoints(point0);
            List<vector> top0_cur = moveBasis(top0, parTop);
            List<vector> bot0_cur = moveBasis(bot0, parBot);
            ITHACAutilities::setIndices2Value(top0_ind, top0_cur, movingIDs, curX);
            ITHACAutilities::setIndices2Value(bot0_ind, bot0_cur, movingIDs, curX);
            ms->setMotion(curX - x0);
            point = ms->curPoints();
            _mesh().movePoints(point);
        }


    /// Perform an Offline solve
    void offlineSolve(word folder = "./ITHACAoutput/Offline/")
    {
        // if the offline solution is already performed read the fields
        if (offline)
        {
            /// Velocity field
            volVectorField& U = _U();
            /// Pressure field
            volScalarField& p = _p();
            /// Energy field
            volScalarField& E = _E();

            ITHACAstream::readMiddleFields(Ufield, U, folder);
            ITHACAstream::readMiddleFields(Efield, E, folder);
            ITHACAstream::readMiddleFields(Pfield, p, folder);
            mu_samples = ITHACAstream::readMatrix("./parsOff_mat.txt");
        }
        // else perform offline stage
        else
        {
            //Vector<double> Uinl(250, 0, 0);
            Vector<double> Uinl(170, 0, 0);

            for (label i = 0; i < mu.rows(); i++)
            {
                //std::cout << "Current mu = " << mu(i, 0) << std::endl;
                //changeViscosity(mu(i, 0));
                updateMesh(mu(i,0),mu(i,1));
                ITHACAstream::writePoints(_mesh().points(), folder, name(i + 1) + "/polyMesh/");
                assignIF(_U(), Uinl);
                truthSolve(folder);

                label j=1;
                word polyMesh2beLinked = folder + name(i+1) + "/" + "polyMesh/";
                while (ITHACAutilities::check_folder(folder + name(i+1) + "/" + name(j)))
                {
                	word folderContLink = folder + name(i+1) + "/" + name(j) + "/";
                	system("ln -s  $(readlink -f " + polyMesh2beLinked + ") " + folderContLink + " >/dev/null 2>&1");
                	j++;

                }
            }
        }
    }

};

int main(int argc, char* argv[])
{
    // Construct the tutorial object  
    tutorial02 example(argc, argv);
    // volScalarField& s = example._p();
    // Info << example._p().boundaryFieldRef()[1] << endl;
    // Info << example._p().boundaryFieldRef()[1][0] << endl;
    // example._p().boundaryFieldRef()[1][0] = 11; 
    // Info << example._p().boundaryFieldRef()[1] << endl;
    // freestreamPressureFvPatchScalarField& Tpatch =
    //     refCast<freestreamPressureFvPatchScalarField>(s.boundaryFieldRef()[1]);
    // scalarField& gradTpatch = Tpatch.freestreamValue();
    // Info << gradTpatch << endl;
    // forAll(gradTpatch, faceI)
    // {
    //     double value = valueList[faceI];
    //     gradTpatch[faceI] = value;
    // }

    //exit(0);

    ITHACAparameters* para = ITHACAparameters::getInstance();

    Eigen::MatrixXd parOff;
    std::ifstream exFileOff("./parsOff_mat.txt");
    if (exFileOff)
    {
        example.mu  = ITHACAstream::readMatrix("./parsOff_mat.txt");
    }

    else
    {
        example.mu.resize(10, 2);
        Eigen::MatrixXd parTop = ITHACAutilities::rand(10, 1, 0, 0.1);
    	Eigen::MatrixXd parBot = ITHACAutilities::rand(10, 1, -0.1, 0);
        example.mu.leftCols(1) = parTop;
        example.mu.rightCols(1) = parBot;
        ITHACAstream::exportMatrix(example.mu , "parsOff", "eigen", "./");
    }

    // Eigen::MatrixXd parOn;
    // std::ifstream exFileOn("./parsOn_mat.txt");
    // if (exFileOn)
    // {
    //     parOn = ITHACAstream::readMatrix("./parsOn_mat.txt");
    // }

    // else
    // {
    //     parOn = ITHACAutilities::rand(20, 1, 1.00e-05, 1.00e-02);
    //     ITHACAstream::exportMatrix(parOn, "parsOn", "eigen", "./");
    // }
    // 

    // Read some parameters from file
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

    exit(0);




    // //Read the lift field
    // ITHACAstream::read_fields(example.liftfield, example._U(), "./lift/");
    // ITHACAutilities::normalizeFields(example.liftfield);
    // // Homogenize the snapshots
    // example.computeLift(example.Ufield, example.liftfield, example.Uomfield);
    // // Perform POD on velocity and pressure and store the first 10 modes
    // ITHACAPOD::getModes(example.Uomfield, example.Umodes, example._U().name(), example.podex, 0, 0,
    //                     NmodesUout);
    // ITHACAPOD::getModes(example.Pfield, example.Pmodes, example._p().name(), example.podex, 0, 0,
    //                     NmodesPout);
    // ITHACAPOD::getModes(example.Efield, example.Emodes, example._E().name(), example.podex, 0, 0,
    //                     NmodesEout);
    
    // // Create the reduced object
    // ReducedCompressibleSteadyNS reduced(example);
    
    // // Reads inlet volocities boundary conditions.
    // word vel_file(para->ITHACAdict->lookup("online_velocities"));
    // Eigen::MatrixXd vel = ITHACAstream::readMatrix(vel_file);

    // //Perform the online solutions
    // for (label k = 0; k < parOn.rows(); k++)
    // {
    //     scalar mu_now = parOn(k, 0);
    //     example.changeViscosity(mu_now);
    //     reduced.setOnlineVelocity(vel);
    //     reduced.projectReducedOperators(NmodesUproj, NmodesPproj, NmodesEproj);
    //     example.restart();
    //     example.turbulence->validate();
    //     reduced.solveOnlineCompressible(mu_now, NmodesUproj, NmodesPproj, NmodesEproj);
    // }

    // if(!ITHACAutilities::check_folder("./ITHACAoutput/checkOff"))
    // {
    //     tutorial02 checkOff(argc, argv);
    //     checkOff.mu  = ITHACAstream::readMatrix("./parsOn_mat.txt");
    //     //Set the inlet boundaries patch 0 directions x and y
    //     checkOff.inletIndex.resize(1, 2);
    //     checkOff.inletIndex(0, 0) = 1;
    //     checkOff.inletIndex(0, 1) = 0;
    //     //Perform the offline solve
    //     checkOff.offline=false;
    //     checkOff.middleExport = false;
    //     checkOff.restart();
    //     checkOff.offlineSolve("./ITHACAoutput/checkOff/");
    // }

    exit(0);
}
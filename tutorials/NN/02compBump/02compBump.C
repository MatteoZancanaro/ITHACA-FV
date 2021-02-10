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

        /// Export intermediate steps
    	middleExport = para->ITHACAdict->lookupOrDefault<bool>("middleExport", true);
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
    ITHACAparameters* para = ITHACAparameters::getInstance();

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

    void updateMesh(double parTop = 0, double parBot = 0)
    {
        _mesh().movePoints(point0);
        if(parTop!=0 || parBot!=0)
        {
        	std::cout << parTop << std::endl;
            List<vector> top0_cur = moveBasis(top0, parTop);
            List<vector> bot0_cur = moveBasis(bot0, parBot);
            ITHACAutilities::setIndices2Value(top0_ind, top0_cur, movingIDs, curX);
            ITHACAutilities::setIndices2Value(bot0_ind, bot0_cur, movingIDs, curX);
            ms->setMotion(curX - x0);
            point = ms->curPoints();
            _mesh().movePoints(point);
    	}
    }

    /// Perform an Offline solve
    void offlineSolve(word folder = "./ITHACAoutput/Offline/")
    {
	    /// Velocity field
        volVectorField& U = _U();
        /// Pressure field
        volScalarField& p = _p();
        /// Energy field
        volScalarField& E = _E();

        // if the offline solution is already performed but POD modes are not present, then read the fields
        if (offline && !ITHACAutilities::check_folder("./ITHACAoutput/POD/1"))
        {
            ITHACAstream::readMiddleFields(Ufield, U, folder);
            ITHACAstream::readMiddleFields(Efield, E, folder);
            ITHACAstream::readMiddleFields(Pfield, p, folder);
            mu_samples = ITHACAstream::readMatrix("./parsOff_mat.txt");
        }
        // if offline stage ha snot been performed, then perform it
        else if (!offline)
        {
            //Vector<double> Uinl(250, 0, 0);
            //Vector<double> Uinl(170, 0, 0);
            double UIFinit = para->ITHACAdict->lookupOrDefault<double>("UIFinit", 170);
            Vector<double> Uinl(UIFinit, 0, 0);

            for (label i = 0; i < mu.rows(); i++)
            {
                //std::cout << "Current mu = " << mu(i, 0) << std::endl;
                //changeViscosity(mu(i, 0));
                updateMesh(mu(i,0),mu(i,1));
                ITHACAstream::writePoints(_mesh().points(), folder, name(i + 1) + "/polyMesh/");
                //assignIF(_U(), Uinl);
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

    // try
    // {
    // 	int i=2;
    // 	if(i<3)
    // 	{
    // 		throw("i piccolo");
    // 	}
    // }

    // catch(const char* message)
    // {
    // 	cerr<<"Succede che "<<message<< endl;
    // }

    // exit(0);

    ITHACAparameters* para = ITHACAparameters::getInstance();

    std::ifstream exFileOff("./parsOff_mat.txt");
    if (exFileOff)
    {
        example.mu  = ITHACAstream::readMatrix("./parsOff_mat.txt");
    }

    else
    {
    	int OffNum = para->ITHACAdict->lookupOrDefault<int>("OffNum", 100);
    	double BumpAmp = para->ITHACAdict->lookupOrDefault<double>("BumpAmp", 0.1);
        example.mu.resize(OffNum, 2);
        Eigen::MatrixXd parTop = ITHACAutilities::rand(example.mu.rows(), 1, 0, BumpAmp);
    	Eigen::MatrixXd parBot = ITHACAutilities::rand(example.mu.rows(), 1, -BumpAmp, 0);
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
    Eigen::MatrixXd parsOn;
    std::ifstream exFileOn("./parsOn_mat.txt");
    if (exFileOn)
    {
        parsOn  = ITHACAstream::readMatrix("./parsOn_mat.txt");
    }

    else
    {
    	int OnNum = para->ITHACAdict->lookupOrDefault<int>("OnNum", 20);
    	double BumpAmp = para->ITHACAdict->lookupOrDefault<double>("BumpAmp", 0.1);
        parsOn.resize(OnNum, 2);
        Eigen::MatrixXd parTopOn = ITHACAutilities::rand(OnNum, 1, 0, BumpAmp);
    	Eigen::MatrixXd parBotOn = ITHACAutilities::rand(OnNum, 1, -BumpAmp, 0);
        parsOn.leftCols(1) = parTopOn;
        parsOn.rightCols(1) = parBotOn;
        ITHACAstream::exportMatrix(parsOn , "parsOn", "eigen", "./");
    }
    // Read some parameters from file
    int NmodesUout = para->ITHACAdict->lookupOrDefault<int>("NmodesUout", 15);
    int NmodesPout = para->ITHACAdict->lookupOrDefault<int>("NmodesPout", 15);
    int NmodesEout = para->ITHACAdict->lookupOrDefault<int>("NmodesEout", 15);
    int NmodesUproj = para->ITHACAdict->lookupOrDefault<int>("NmodesUproj", 10);
    int NmodesPproj = para->ITHACAdict->lookupOrDefault<int>("NmodesPproj", 10);
    int NmodesEproj = para->ITHACAdict->lookupOrDefault<int>("NmodesEproj", 10);
    //Set the inlet boundaries patch 0 directions x and y
    label idInl = example._mesh().boundaryMesh().findPatchID("inlet");
    example.inletIndex.resize(1, 2);
    example.inletIndex(0, 0) = idInl;
    example.inletIndex(0, 1) = 0;
    //Perform the offline solve
    example.offlineSolve();
    //exit(0);




    //Read the lift field
    ITHACAstream::read_fields(example.liftfield, example._U(), "./lift/");
    ITHACAutilities::normalizeFields(example.liftfield);

    if(!ITHACAutilities::check_folder("./ITHACAoutput/POD/1")) // If POD has already been performed, you do not have to omogenize anything
    {
	    // Homogenize the snapshots
	    example.computeLift(example.Ufield, example.liftfield, example.Uomfield);
	}
    
    // Perform POD on velocity and pressure and store the first 10 modes
    example.updateMesh(); // Move the mesh to the original geometry to get the modes into a mid mesh

    ITHACAPOD::getModes(example.Uomfield, example.Umodes, example._U().name(), example.podex, 0, 0,
                        NmodesUout);
    ITHACAPOD::getModes(example.Pfield, example.Pmodes, example._p().name(), example.podex, 0, 0,
                        NmodesPout);
    ITHACAPOD::getModes(example.Efield, example.Emodes, example._E().name(), example.podex, 0, 0,
                        NmodesEout);
    
    // Create the reduced object
    ReducedCompressibleSteadyNS reduced(example);
    
    // // Reads inlet volocities boundary conditions.
    // word vel_file(para->ITHACAdict->lookup("online_velocities"));
    // Eigen::MatrixXd vel = ITHACAstream::readMatrix(vel_file);
    // 
    
    Eigen::MatrixXd vel(1,1);
    vel(0,0) = example._U().boundaryFieldRef()[idInl][0][0];

    //Perform the online solutions
    for (label k = 0; k < parsOn.rows(); k++)
    {
        //scalar mu_now = parOn(k, 0);
        //example.changeViscosity(mu_now);
        example.updateMesh(parsOn(k,0), parsOn(k,1));
        ITHACAstream::writePoints(example._mesh().points(), "./ITHACAoutput/Online/", name(k + 1) + "/polyMesh/");
        reduced.setOnlineVelocity(vel);
        reduced.projectReducedOperators(NmodesUproj, NmodesPproj, NmodesEproj);
        example.restart();
        example.turbulence->validate();
        reduced.solveOnlineCompressible(NmodesUproj, NmodesPproj, NmodesEproj);
    }

    if(!ITHACAutilities::check_folder("./ITHACAoutput/checkOff"))
    {
        tutorial02 checkOff(argc, argv);
        checkOff.mu  = ITHACAstream::readMatrix("./parsOn_mat.txt");
        //Set the inlet boundaries patch 0 directions x and y
        label idInl = checkOff._mesh().boundaryMesh().findPatchID("inlet");
        checkOff.inletIndex.resize(1, 2);
        checkOff.inletIndex(0, 0) = idInl;
        checkOff.inletIndex(0, 1) = 0;
        //Perform the offline solve
        checkOff.offline = false;
        checkOff.middleExport = false;
        checkOff.restart();
        checkOff.offlineSolve("./ITHACAoutput/checkOff/");

        PtrList<volVectorField> onlineU;
        PtrList<volScalarField> onlineP;
        PtrList<volScalarField> onlineE;
        ITHACAstream::read_fields(onlineU,checkOff._U(),"./ITHACAoutput/Online/");
        ITHACAstream::read_fields(onlineP,checkOff._p(),"./ITHACAoutput/Online/");
        ITHACAstream::read_fields(onlineE,checkOff._E(),"./ITHACAoutput/Online/");
        Eigen::MatrixXd errorU = ITHACAutilities::errorL2Rel(checkOff.Ufield,
                            onlineU);
        Eigen::MatrixXd errorP = ITHACAutilities::errorL2Rel(checkOff.Pfield,
                            onlineP);
        Eigen::MatrixXd errorE = ITHACAutilities::errorL2Rel(checkOff.Efield,
                            onlineE);
        ITHACAstream::exportMatrix(errorU,"errorU", "python", "./ITHACAoutput/checkOff/");
        ITHACAstream::exportMatrix(errorP,"errorP", "python", "./ITHACAoutput/checkOff/");
        ITHACAstream::exportMatrix(errorE,"errorE", "python", "./ITHACAoutput/checkOff/");

        for(label j=0; j<checkOff.mu.rows(); j++)
        {
        	volVectorField Ue = checkOff.Ufield[j] - onlineU[j];
        	volScalarField Pe = checkOff.Pfield[j] - onlineP[j];
        	volScalarField Ee = checkOff.Efield[j] - onlineE[j];
        	Ue.rename("Ue");
        	Pe.rename("Pe");
        	Ee.rename("Ee");
        	ITHACAstream::exportSolution(Ue, name(j+1), "./ITHACAoutput/checkOff/");
	    	ITHACAstream::exportSolution(Pe, name(j+1), "./ITHACAoutput/checkOff/");
	    	ITHACAstream::exportSolution(Ee, name(j+1), "./ITHACAoutput/checkOff/");
        }
    }

    exit(0);
}
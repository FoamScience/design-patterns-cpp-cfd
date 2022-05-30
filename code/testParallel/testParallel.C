/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2022 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    testParallel

Description

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "label.H"
#include "ops.H"
#include "mpi.h"
#include "processorPolyPatch.H"
#include <chrono>
typedef std::chrono::high_resolution_clock CTime;
typedef std::chrono::milliseconds ms;
typedef std::chrono::duration<float> fsec;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

List<bool> gatherTest() {
    List<bool> localLst(Pstream::nProcs(), false);
    localLst[Pstream::myProcNo()] =  true;
    Pstream::gatherList(localLst);
    return localLst;
}

bool gatherTest2() {
    bool t = false;
    if (Pstream::master()){
        t = true;
    }
    Pstream::gather(t, orOp<label>());
    return t;
}

List<label> scatterTest() {
    List<label> localLst(Pstream::nProcs(), -1);
    if (Pstream::master()){
        forAll(localLst, ei) { localLst[ei] =  ei; }
    }
    Pstream::scatterList(localLst);
    return localLst;
}

bool scatterTest2() {
    bool sc = false;
    if (Pstream::myProcNo() == 2){
        sc = true;
    }
    Pstream::scatter(sc);
    return sc;
}

label reduceTest() {
    label localVar = -1;
    Foam::reduce(localVar, sumOp<decltype(localVar)>());
    return localVar;
}

label reduceTest2() {
    return Foam::returnReduce(-1, sumOp<label>());
}

void endlessLoopCollectives(label globalNCells) {
  // Does some calculations until cell count reaches global desired nCells
  label currentNCells = 0;
  do
  {
    // Perform calculations on all processors
    currentNCells += 1;

    reduce(currentNCells, sumOp<label>());
    // On some condition, a processor should not continue, and immediately
    // returns control to the caller
    if (Pstream::myProcNo() == 1) return;

    // !!! who's going to reduce this!
  } while (currentNCells < globalNCells);
  return;
}

struct Edge;
Istream&  operator>>(Istream& is, Edge& e);

struct Edge {
    label destination;
    scalar weight;
    bool operator==(const Edge& ej) const {
        return destination == ej.destination;
    }
    bool operator!=(const Edge& ej) const {
        return destination != ej.destination;
    }
};

Ostream&  operator<<(Ostream& os, const Edge& e) {
    os << e.destination << " " << e.weight;
    return os;
}

Istream&  operator>>(Istream& is, Edge& e) {
    is >> e.destination;
    is >> e.weight;
    return is;
}

using Graph = List<List<Edge>>;

Edge testEdgeScatter() {
    Edge ej;
    if (Pstream::master())
    {
        ej.destination = 16;
        ej.weight = 5.2;
    }

    Pstream::scatter(ej);
    return ej;
}

Graph testGraph(const fvMesh& mesh) {
    Graph g(Pstream::nProcs());
    forAll(mesh.boundaryMesh(), pi) {
        auto* patch = dynamic_cast<processorPolyPatch*>(const_cast<polyPatch*>(&mesh.boundaryMesh()[pi]));
        if (patch) {
            Edge e;
            e.destination = patch->neighbProcNo();
            g[Pstream::myProcNo()].append(e);
        }
    }
    Pstream::gatherList(g);
    Pstream::scatterList(g);
    return g;
}


void testNonBlockingComms(const fvMesh& mesh) {
    label NITERS = 20;
    auto t0 = CTime::now();
    labelList dt(4000000);
    Info << "Sending roughly " << (sizeof(labelList) + (sizeof(int) * dt.size()))*1e-6 << " Mbytes of data " << NITERS << " times." << endl;
    std::generate(dt.begin(), dt.end(), [n = 0] () mutable { return n++; });

    double start_time  = 0;
    double end_time  = 0;
    double acc_wait_time = 0;
    double acc_calc_time = 0;
    const auto& patches = mesh.boundaryMesh();
    for (size_t i = 0; i < NITERS; i++) {
        start_time = MPI_Wtime();
        PstreamBuffers pBufs(Pstream::commsTypes::nonBlocking);

        // Send
        forAll(patches, patchi)
        {
          const auto *p = dynamic_cast<processorPolyPatch const*>(&patches[patchi]);
            if (p) {
                UOPstream toNeighb(p->neighbProcNo(), pBufs);
                toNeighb << dt;
            }
        }

        // Overlapped unrelated calculations
        pBufs.finishedSends(); // <- Calls Pstream::waitRequests

        // Receive
        forAll(patches, patchi)
        {
          const auto *p = dynamic_cast<processorPolyPatch const*>(&patches[patchi]);
            if (p) {
                UIPstream fromNb(p->neighbProcNo(), pBufs);
                labelList nbrDt(fromNb);
            }
        }
        end_time = MPI_Wtime();
        acc_wait_time +=  end_time-start_time;
        auto tc0 = CTime::now();
        for(int ii=0; auto&& p : dt) { 
            ii++;
            void(Foam::pow(p%90000, 2+ii%9)+Foam::factorial(p%90000)/Foam::pow(p,2+ii%12));
        }
        auto tc1 = CTime::now();
        fsec fs = tc1 - tc0;
        acc_calc_time += fs.count();

    }

    auto t1 = CTime::now();
    fsec fs = t1 - t0;
    Pout << fs.count()/NITERS << " " << acc_wait_time/NITERS << " " << acc_calc_time/NITERS << endl;
}

void testBlockingComms(const fvMesh& mesh) {
    label NITERS = 20;
    auto t0 = CTime::now();
    labelList dt(4000000);
    Info << "Sending roughly " << (sizeof(labelList) + (sizeof(int) * dt.size()))*1e-6 << " Mbytes of data " << NITERS << " times." << endl;
    std::generate(dt.begin(), dt.end(), [n = 0] () mutable { return n++; });

    double start_time  = 0;
    double end_time  = 0;
    double acc_wait_time = 0;
    double acc_calc_time = 0;
    const auto& patches = mesh.boundaryMesh();
    for (size_t i = 0; i < NITERS; i++) {
        start_time = MPI_Wtime();

        // Send
        forAll(patches, patchi)
        {
          const auto *p = dynamic_cast<processorPolyPatch const*>(&patches[patchi]);
            if (p) {
                OPstream toNeighb(Pstream::commsTypes::blocking, p->neighbProcNo());
                toNeighb << dt;
            }
        }

        // Receive
        forAll(patches, patchi)
        {
          const auto *p = dynamic_cast<processorPolyPatch const*>(&patches[patchi]);
            if (p) {
                IPstream fromNb(Pstream::commsTypes::blocking, p->neighbProcNo());
                labelList nbrDt(fromNb);
            }
        }
        end_time = MPI_Wtime();
        acc_wait_time +=  end_time-start_time;
        auto tc0 = CTime::now();
        for(int ii=0; auto&& p : dt) { 
            ii++;
            void(Foam::pow(p%90000, 2+ii%9)+Foam::factorial(p%90000)/Foam::pow(p,2+ii%12));
        }
        auto tc1 = CTime::now();
        fsec fs = tc1 - tc0;
        acc_calc_time += fs.count();

    }

    auto t1 = CTime::now();
    fsec fs = t1 - t0;
    Pout << fs.count()/NITERS << " " << acc_wait_time/NITERS << " " << acc_calc_time/NITERS << endl;
}

int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    testNonBlockingComms(mesh);
    testBlockingComms(mesh);

    Info<< nl << "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
        << "  ClockTime = " << runTime.elapsedClockTime() << " s"
        << nl << endl;

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //

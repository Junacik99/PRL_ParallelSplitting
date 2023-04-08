/******************
*  Martin Takacs  *
*    xtakac07     *
*   PRL project   *
*  Parallel Split *
******************/

#include <iostream>
#include <mpi.h>
#include <vector>
#include <fstream>
#include <string>


int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);

	// Read input from a file
	const std::string filename = "numbers";
	std::ifstream file(filename);
	std::string line;
	std::getline(file, line);

	// Init buffer
	std::vector<int> sendbuf;

	// Convert to numbers
	for (char c : line) {
		sendbuf.push_back(static_cast<unsigned char>(c));
	}

	int n = sendbuf.size();

	int rank, size;
	// Choose median
	int m = (n/2) - 1;
		
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	

	// Send m to all processes via broadcast
	// m is a splitting point
	int median = sendbuf[m];
	if (rank == 0) std::cout << "median = " << median << std::endl;
	MPI_Bcast(&median, 1, MPI_BYTE, 0, MPI_COMM_WORLD);

	// Divide sequence into N sequences for every process
	int subArrSize = n / size;
	std::vector<int> recvbuf(subArrSize);

	// Scatter the data from the root process to all other processes
	MPI_Scatter(sendbuf.data(), subArrSize, MPI_INT, recvbuf.data(), subArrSize, MPI_INT, 0, MPI_COMM_WORLD);

	// Every process Pi divide its sequence to Li, Ei, Gi
	std::vector<int> li, ei, gi;
	for (int i = 0; i < subArrSize; ++i) {
		int x = recvbuf[i];
		if (x < median)
			li.push_back(x);
		else if (x > median)
			gi.push_back(x);
		else
			ei.push_back(x);
	}

	// Calculate offsets and sizes
	std::vector<int> 
		loffsets(size), lcounts(size),
		eoffsets(size), ecounts(size),
		goffsets(size), gcounts(size);

	int lai = li.size(), lzi,
		eai = ei.size(), ezi,
		gai = gi.size(), gzi;

	if (rank == 0) {
		lzi = 0; 
		ezi = 0;
		gzi = 0;
	}

	MPI_Exscan(&lai, &lzi, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Gather(&lai, 1, MPI_INT, lcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gather(&lzi, 1, MPI_INT, loffsets.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Exscan(&eai, &ezi, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Gather(&eai, 1, MPI_INT, ecounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gather(&ezi, 1, MPI_INT, eoffsets.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Exscan(&gai, &gzi, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Gather(&gai, 1, MPI_INT, gcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gather(&gzi, 1, MPI_INT, goffsets.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

	// Calculate sums
	int lsize = 0, esize = 0, gsize = 0;
	if (rank == 0) {
		for (int x : lcounts)
			lsize += x;
		for (int x : ecounts)
			esize += x;
		for (int x : gcounts)
			gsize += x;
	}
	

	// Gather to final vectors
	std::vector<int> less(lsize), equal(esize), greater(gsize);
	MPI_Gatherv(li.data(), lai, MPI_INT, less.data(), lcounts.data(), loffsets.data(), MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gatherv(ei.data(), eai, MPI_INT, equal.data(), ecounts.data(), eoffsets.data(), MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gatherv(gi.data(), gai, MPI_INT, greater.data(), gcounts.data(), goffsets.data(), MPI_INT, 0, MPI_COMM_WORLD);

	
	// Print result
	if (rank == 0) {
		std::cout << "Input array = [ ";
		for (int x : sendbuf)
			std::cout << x << " ";
		std::cout << "]\n";
		std::cout << "Median = " << median << std::endl;
		std::cout << "L = [ ";
		for (int x : less)
			std::cout << x << " ";
		std::cout << "]\n";
		std::cout << "E = [ ";
		for (int x : equal)
			std::cout << x << " ";
		std::cout << "]\n";
		std::cout << "G = [ ";
		for (int x : greater)
			std::cout << x << " ";
		std::cout << "]\n";
	}


	MPI_Finalize();
	return 0;
}

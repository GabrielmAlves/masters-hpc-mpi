#include <cblas.h>
#include <iostream>
#include <vector>

int main() {
	const int M = 2, N = 2, K = 3;
	std::vector<float> A = {1, 2, 3,
				4, 5, 6};
	std::vector<float> B = {7, 8,
				9, 10,
				11, 12};
	std::vector<float> C(M*N, 0.0f);

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		    M, N, K,
		    1.0f, A.data(), K,
		    	  B.data(), N,
		    0.0f, C.data(), N);
	std::cout << "C = A X B: " << std::endl;

	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			std::cout << C[i*N + j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << "Threads do OpenBLAS: " << openblas_get_num_threads() << std::endl;

	return 0;
}

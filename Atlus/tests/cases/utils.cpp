#include "../headers/utils.h"
#include "../../src/headers/utils.h"

std::vector<std::vector<double>> classificationLinearMultipleX() {
	std::vector<std::vector<double>> m2;

	srand(time(NULL));
	for (int i = 0; i < 100; i++) {
		m2.push_back({ random() * 0.9 + 1.0, random() * 0.9 + 1.0 });
	}
	for (int i = 0; i < 100; i++) {
		m2.push_back({ random() * 0.9 + 2.0, random() * 0.9 + 2.0 });
	}

	//std::cout << "TEST TEST TEST TEST " << m2[0][0] << std::endl;

	return m2;
}

std::vector<std::vector<double>> classificationLinearMultipleY() {
	std::vector<std::vector<double>> m2;

	for (int i = 0; i < 50; i++) {
		m2.push_back({ 1 });
	}
	for (int i = 0; i < 50; i++) {
		m2.push_back({ -1 });
	}

	return m2;
}
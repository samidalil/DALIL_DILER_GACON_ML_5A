#include "../headers/utils.h"
#include "../../src/headers/utils.h"

std::vector<std::vector<double>> classificationLinearMultipleX() {
	std::vector<std::vector<double>> m2;

	srand(time(NULL));
	for (int i = 0; i < 50; i++) {
		m2.push_back({ random() * 0.9 + 1.0, random() * 0.9 + 1.0 });
	}
	for (int i = 0; i < 50; i++) {
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

std::vector<std::vector<double>> classificationCrossX() {
	std::vector<std::vector<double>> m2;

	srand(time(NULL));
	for (int i = 0; i < 500; i++) {
		m2.push_back({ random() * 2.0 - 1.0, random() * 2.0 - 1.0 });
	}

	return m2;
}

std::vector<std::vector<double>> classificationCrossY() {
	srand(time(NULL));

	std::vector<std::vector<double>> X;
	for (int i = 0; i < 500; i++) {
		X.push_back({ random() * 2.0 - 1.0, random() * 2.0 - 1.0 });
	}

	std::vector<std::vector<double>> m2;

	srand(time(NULL));
	for (int i = 0; i < 500; i++) {
		if(abs(X[i][0])<=0.3 || abs(X[i][1]) <=0.3)
			m2.push_back({1});
		else
			m2.push_back({-1});
	}

	//std::cout << "TEST TEST TEST TEST " << m2[0][0] << std::endl;

	return m2;
}
#include "../headers/utils.h"

std::vector<std::vector<double>> classificationLinearMultipleX() {
	std::vector<std::vector<double>> m2;

	/*for (int i = 0; i < 50; i++) {
		m2.push_back({ 1 });
	}
	for (int i = 0; i < 50; i++) {
		m2.push_back({ -1 });
	}*/
	m2.push_back({ -1, 1 });

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
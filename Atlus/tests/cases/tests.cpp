#include "../headers/framework.h"

#include "handler.cpp"

#include <iostream>

TEST_METHOD(linearSimple)
{
	MLPHandler handler({ 2, 1 });

	m2 inputs = { { 1, 1 }, { 2, 3 }, {3, 3} };
	m2 outputs = { { 1 }, { -1 }, { -1 } };

	handler.trainClassification(inputs, outputs, 0.05, 10000);

	std::cout << handler.evaluate(inputs, outputs) << std::endl;
}
#include "../headers/framework.h"

#include "handler.cpp"
#include "../../src/headers/utils.h"

#include <iostream>

std::vector<uint> range(uint size)
{
	return std::vector<uint>(size);
}

constexpr double LEARNING_RATE = 0.0005;
constexpr uint EPOCHS = 100000;

TEST_METHOD(classificationLinearSimple)
{
	MLPHandler handler({ 2, 1 });

	m2 inputs = { { 1, 1 }, { 2, 3 }, {3, 3} };
	m2 outputs = { { 1 }, { -1 }, { -1 } };

	handler.trainClassification(inputs, outputs, LEARNING_RATE, EPOCHS);
	handler.evaluateClassification(inputs, outputs);
}

TEST_METHOD(classificationLinearMultiple)
{
	MLPHandler handler({ 2, 1 });

	m2 inputs;
	m2 outputs;
	
	srand(time(NULL));

	for (auto i : range(50)) inputs.push_back({ random() * 0.9 + 1.0, random() * 0.9 + 1.0 });
	for (auto i : range(50)) inputs.push_back({ random() * 0.9 + 2.0, random() * 0.9 + 2.0 });

	for (auto i : range(50)) outputs.push_back({ 1 });
	for (auto i : range(50)) outputs.push_back({ -1 });

	handler.trainClassification(inputs, outputs, LEARNING_RATE, EPOCHS);
	handler.evaluateClassification(inputs, outputs);
}

TEST_METHOD(classificationXOR)
{
	MLPHandler handler({ 2, 1 });

	m2 inputs = { { 1, 0 }, { 0, 1 }, {0, 0}, {1, 1} };
	m2 outputs = { { 1 }, { -1 }, { -1 }, {-1} };

	handler.trainClassification(inputs, outputs, LEARNING_RATE, EPOCHS);
	handler.evaluateClassification(inputs, outputs);
}

TEST_METHOD(classificationCross)
{
	MLPHandler handler({ 2, 4, 1 });
	
	m2 inputs;
	m2 outputs;

	srand(time(NULL));

	for (auto i : range(500)) inputs.push_back({ random() * 2.0 - 1.0, random() * 2.0 - 1.0 });
	for (auto p : inputs)
	{
		if (abs(p[0]) <= 0.3 || abs(p[1]) <= 0.3)
			outputs.push_back({ 1 });
		else
			outputs.push_back({ -1 });
	}

	handler.trainClassification(inputs, outputs, LEARNING_RATE, EPOCHS);
	handler.evaluateClassification(inputs, outputs);
}

TEST_METHOD(classificationMultiLinear3Classes)
{
	MLPHandler handler({ 2, 3 });

	m2 inputs;
	m2 outputs;

	srand(time(NULL));

	for (auto i : range(1000)) inputs.push_back({ random() * 2.0 - 1.0, random() * 2.0 - 1.0 });
	for (auto p : inputs)
	{
		double a = -p[0] - p[1] - 0.5;
		double b = p[0] - p[1] - 0.5;

		if (a > 0 && p[1] < 0 && b < 0)
			outputs.push_back({ 1, 0, 0 });
		else if (a < 0 && p[1] > 0 && b < 0)
			outputs.push_back({ 0, 1, 0 });
		else if (a < 0 && p[1] < 0 && b > 0)
			outputs.push_back({ 0, 0, 1 });
		else
			outputs.push_back({ 0, 0, 0 });
	}

	handler.trainClassification(inputs, outputs, LEARNING_RATE, EPOCHS);
	handler.evaluateClassification(inputs, outputs);
}

TEST_METHOD(classificationMultiCross)
{
	MLPHandler handler({ 2, 4, 5, 3 });

	m2 inputs;
	m2 outputs;

	srand(time(NULL));

	for (auto i : range(1000)) inputs.push_back({ random() * 2.0 - 1.0, random() * 2.0 - 1.0 });
	for (auto p : inputs)
	{
		if (abs(fmod(p[0], 0.5)) <= 0.25 && abs(fmod(p[1], 0.5)) > 0.25)
			outputs.push_back({ 1, 0, 0 });
		else if (abs(fmod(p[0], 0.5)) > 0.25 && abs(fmod(p[1], 0.5)) <= 0.25)
			outputs.push_back({ 0, 1, 0 });
		else
			outputs.push_back({ 0, 0, 1 });
	}

	handler.trainClassification(inputs, outputs, LEARNING_RATE, EPOCHS);
	handler.evaluateClassification(inputs, outputs);
}

TEST_METHOD(regressionLinearSimple2D)
{
	MLPHandler handler({ 1, 1 });

	m2 inputs = { {1}, {2} };
	m2 outputs = { { 2 }, { 3 } };

	handler.trainRegression(inputs, outputs, LEARNING_RATE, EPOCHS);
	handler.evaluateRegression(inputs, outputs);
}

TEST_METHOD(regressionLinearNonSimple2D)
{
	MLPHandler handler({ 1, 1, 1 }); //?

	m2 inputs = { {1}, {2}, {3} };
	m2 outputs = { { 2 }, { 3 }, { 2.5 } };

	handler.trainRegression(inputs, outputs, LEARNING_RATE, EPOCHS);
	handler.evaluateRegression(inputs, outputs);
}

TEST_METHOD(regressionLinearSimple3D)
{
	MLPHandler handler({ 2, 1 });

	m2 inputs = { {1,1}, {2,2}, {3,1} };
	m2 outputs = { { 2 }, { 3 }, {2.5} };

	handler.trainRegression(inputs, outputs, LEARNING_RATE, EPOCHS);
	handler.evaluateRegression(inputs, outputs);
}

TEST_METHOD(regressionLinearTricky3D)
{
	MLPHandler handler({ 2, 1 });

	m2 inputs = { {1,1}, {2,2}, {3,3} };
	m2 outputs = { { 1 }, { 2 }, {3} };

	handler.trainRegression(inputs, outputs, LEARNING_RATE, EPOCHS);
	handler.evaluateRegression(inputs, outputs);
}

TEST_METHOD(regressionNonLinearSimple3D)
{
	MLPHandler handler({ 2, 2, 1 });

	m2 inputs = { {1,0}, {0,1}, {1,1}, {0,0} };
	m2 outputs = { { 2 }, { 1 }, {-2}, {-1} };

	handler.trainRegression(inputs, outputs, LEARNING_RATE, EPOCHS);
	handler.evaluateRegression(inputs, outputs);
}
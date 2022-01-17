#include "../headers/framework.h"
#include "../headers/utils.h"

#include "handler.cpp"

#include <iostream>

TEST_METHOD(classificationLinearSimple)
{
	MLPHandler handler({ 2, 1 });

	m2 inputs = { { 1, 1 }, { 2, 3 }, {3, 3} };
	m2 outputs = { { 1 }, { -1 }, { -1 } };

	handler.trainClassification(inputs, outputs, 0.05, 10000);
	handler.evaluateClassification(inputs, outputs);
}

TEST_METHOD(classificationLinearMultiple)
{
	MLPHandler handler({ 2, 1 });

	m2 inputs;
	m2 outputs;
	
	srand(time(NULL));

	for (int i = 0; i < 50; i++) inputs.push_back({ random() * 0.9 + 1.0, random() * 0.9 + 1.0 });
	for (int i = 0; i < 50; i++) inputs.push_back({ random() * 0.9 + 2.0, random() * 0.9 + 2.0 });

	for (int i = 0; i < 50; i++) outputs.push_back({ 1 });
	for (int i = 0; i < 50; i++) outputs.push_back({ -1 });

	handler.trainClassification(inputs, outputs, 0.05, 10000);
	handler.evaluateClassification(inputs, outputs);
}

TEST_METHOD(classificationXOR)
{
	MLPHandler handler({ 2, 1 });

	m2 inputs = { { 1, 0 }, { 0, 1 }, {0, 0}, {1, 1} };
	m2 outputs = { { 1 }, { -1 }, { -1 }, {-1} };

	handler.trainClassification(inputs, outputs, 0.05, 10000);
	handler.evaluateClassification(inputs, outputs);
}

TEST_METHOD(classificationCross)
{
	MLPHandler handler({ 2, 1 });

	m2 inputs = classificationCrossX();
	m2 outputs = classificationCrossY();

	handler.trainClassification(inputs, outputs, 0.05, 10000);
	handler.evaluateClassification(inputs, outputs);

}

TEST_METHOD(classificationMultiLinear3Classes)
{

}

TEST_METHOD(classificationMultiCross)
{

}

TEST_METHOD(regressionLinearSimple2D)
{
	MLPHandler handler({ 1, 1 });

	m2 inputs = { {1}, {2} };
	m2 outputs = { { 2 }, { 3 } };

	handler.trainRegression(inputs, outputs, 0.05, 10000);
	handler.evaluateRegression(inputs, outputs);
}

TEST_METHOD(regressionLinearNonSimple2D)
{
	MLPHandler handler({ 1, 1, 1 }); //?

	m2 inputs = { {1}, {2}, {3} };
	m2 outputs = { { 2 }, { 3 }, { 2.5 } };

	handler.trainRegression(inputs, outputs, 0.05, 10000);
	handler.evaluateRegression(inputs, outputs);
}

TEST_METHOD(regressionLinearSimple3D)
{
	MLPHandler handler({ 2, 1 });

	m2 inputs = { {1,1}, {2,2}, {3,1} };
	m2 outputs = { { 2 }, { 3 }, {2.5} };

	handler.trainRegression(inputs, outputs, 0.05, 10000);
	handler.evaluateRegression(inputs, outputs);
}

TEST_METHOD(regressionLinearTricky3D)
{
	MLPHandler handler({ 2, 1 });

	m2 inputs = { {1,1}, {2,2}, {3,3} };
	m2 outputs = { { 1 }, { 2 }, {3} };

	handler.trainRegression(inputs, outputs, 0.05, 10000);
	handler.evaluateRegression(inputs, outputs);
}

TEST_METHOD(regressionNonLinearSimple3D)
{
	MLPHandler handler({ 2, 2, 1 });

	m2 inputs = { {1,0}, {0,1}, {1,1}, {0,0} };
	m2 outputs = { { 2 }, { 1 }, {-2}, {-1} };

	handler.trainRegression(inputs, outputs, 0.05, 10000);
	handler.evaluateRegression(inputs, outputs);
}
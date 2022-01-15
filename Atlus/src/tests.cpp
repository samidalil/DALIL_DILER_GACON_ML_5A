#include "../headers/core.h"
#include <iostream>

// Tests

// TD: à refaire
class MLPTestClass
{
protected:
	MLPData* model;
	double* samplesInputs;
	double* samplesExpectedOutputs;

	static double* flatSamples(m2 samples)
	{
		uint sampleCount = samples.size();
		uint sampleDim = samples[0].size();
		double* data = new double[sampleCount * sampleDim];

		for (uint i = 0; i < sampleCount; ++i)
			std::copy(samples[i].begin(), samples[i].end(), data + sampleDim * i);

		return data;
	}

	MLPTestClass(std::vector<uint> npl, m2 samplesInputs, m2 samplesExpectedOutputs)
	{
		uint* nplArray = npl.data();

		this->model = createMlpModel(nplArray, npl.size());
		this->samplesInputs = MLPTestClass::flatSamples(samplesInputs);
		this->samplesExpectedOutputs = MLPTestClass::flatSamples(samplesExpectedOutputs);
	}

	~MLPTestClass()
	{
		delete this->model;
		delete[] this->samplesInputs;
		delete[] this->samplesExpectedOutputs;
	}
public:
	virtual void invoke()
	{
	}
};

class LinearSimple : public MLPTestClass
{
public:
	LinearSimple() : MLPTestClass(
		{ 2, 1 },
		{ { 1, 1 }, { 2, 3 }, { 3, 3 } },
		{ { 1 }, { -1 }, { -1 } }
	)
	{
	}

	void invoke() override
	{
		trainMlpModelClassification(this->model, this->samplesInputs, this->samplesExpectedOutputs, 3, 2, 1, 0.05, 20000);
		double* result = predictMlpModelClassification(model, samplesInputs);

		std::cout << "Accuracy: " << evaluateModelAccuracy(this->model, this->samplesInputs, this->samplesExpectedOutputs, 3, 2, 1) << std::endl;

		for (int i = 0; i < 3; ++i)
		{
			std::cout << "Input: " << this->samplesInputs[i * 2] << ", " << this->samplesInputs[i * 2 + 1] << std::endl;
			std::cout << "Output: " << result[i] << std::endl;
			std::cout << "Expected output: " << this->samplesExpectedOutputs[i] << std::endl;
		}
	}
};

void main()
{
	LinearSimple test;

	test.invoke();
}

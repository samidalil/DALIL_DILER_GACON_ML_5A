#include "../../src/headers/core.h"

#include <vector>
#include <iterator>

struct MLPHandler
{
	static double* flatSamples(m2 samples)
	{
		uint sampleCount = samples.size();
		uint sampleDim = samples[0].size();
		double* data = new double[sampleCount * sampleDim];

		for (uint i = 0; i < sampleCount; ++i)
			std::copy(samples[i].begin(), samples[i].end(), data + sampleDim * i);

		return data;
	}

	MLPData* model;
	uint outputDim;

	MLPHandler(std::vector<uint> npl)
	{
		const uint size = npl.size();
		this->model = createMlpModel(npl.data(), size);
		this->outputDim = npl[size - 1];
	}

	~MLPHandler()
	{
		delete this->model;
	}

	double evaluate(m2 inputs, m2 outputs)
	{
		const double* inputArr = flatSamples(inputs);
		const double* outputArr = flatSamples(outputs);

		const double result = evaluateModelAccuracy(this->model, inputArr, outputArr, inputs.size(), inputs[0].size(), outputs[0].size());

		delete[] inputArr;
		delete[] outputArr;

		return result * 100;
	}

	m1 predictClassification(m1 input)
	{
		const double* resultArr = predictMlpModelClassification(this->model, input.data());
		m1 result;

		result.reserve(this->outputDim);

		std::copy(resultArr, resultArr + this->outputDim, std::back_inserter(result));

		delete[] resultArr;

		return result;
	}

	m1 predictRegression(m1 input)
	{
		const double* resultArr = predictMlpModelRegression(this->model, input.data());
		m1 result;

		result.reserve(this->outputDim);

		std::copy(resultArr, resultArr + this->outputDim, std::back_inserter(result));

		delete[] resultArr;

		return result;
	}

	void trainClassification(m2 inputs, m2 outputs, double alpha, uint epochs)
	{
		const double* inputArr = flatSamples(inputs);
		const double* outputArr = flatSamples(outputs);

		trainMlpModelClassification(this->model, inputArr, outputArr, inputs.size(), inputs[0].size(), outputs[0].size(), alpha, epochs);

		delete[] inputArr;
		delete[] outputArr;
	}

	void trainRegression(m2 inputs, m2 outputs, double alpha, uint epochs)
	{
		const double* inputArr = flatSamples(inputs);
		const double* outputArr = flatSamples(outputs);

		trainMlpModelRegression(this->model, inputArr, outputArr, inputs.size(), inputs[0].size(), outputs[0].size(), alpha, epochs);

		delete[] inputArr;
		delete[] outputArr;
	}
};
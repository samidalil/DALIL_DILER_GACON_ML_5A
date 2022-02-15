#include "../../src/headers/core.h"

#include <vector>
#include <iterator>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

struct MLPHandler
{
public:
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

	MLPHandler(const char* path)
	{
		std::ifstream file(path);
		std::stringstream buffer;

		buffer << file.rdbuf();

		file.close();

		this->model = deserializeModel(buffer.str());
		this->outputDim = this->model->npl[this->model->L];
	}

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

	double evaluateClassification(m2 inputs, m2 outputs)
	{
		const double* inputArr = flatSamples(inputs);
		const double* outputArr = flatSamples(outputs);

		const double result = evaluateModelAccuracyClassification(this->model, inputArr, outputArr, inputs.size(), inputs[0].size(), outputs[0].size()) * 100;

		delete[] inputArr;
		delete[] outputArr;

		std::cout << "Accuracy : " << result << "%" << std::endl;

		return result;
	}

	double evaluateRegression(m2 inputs, m2 outputs)
	{
		const double* inputArr = flatSamples(inputs);
		const double* outputArr = flatSamples(outputs);

		const double result = evaluateModelAccuracyRegression(this->model, inputArr, outputArr, inputs.size(), inputs[0].size(), outputs[0].size()) * 100;

		delete[] inputArr;
		delete[] outputArr;

		std::cout << "Accuracy : " << result << "%" << std::endl;

		return result;
	}

	double evaluateRBF(m2 inputs, m2 outputs)
	{
		const double* inputArr = flatSamples(inputs);
		const double* outputArr = flatSamples(outputs);

		const double result = evaluateModelAccuracyRBF(this->model, inputArr, outputArr, inputs.size(), inputs[0].size(), outputs[0].size()) * 100;

		delete[] inputArr;
		delete[] outputArr;

		std::cout << "Accuracy : " << result << "%" << std::endl;

		return result;
	}

	m1 predictClassification(m1 input)
	{
		const double* resultArr = predictMlpModelClassification(this->model, input.data());
		m1 result;

		result.reserve(this->outputDim);

		std::copy(resultArr, resultArr + this->outputDim, std::back_inserter(result));

		delete[] resultArr;

		std::cout << "Predict : " << result[0] << std::endl;

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

	void save(std::string path)
	{
		std::ofstream file(path);

		file << serializeModel(this->model);

		file.close();
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

	void trainRBF(m2 inputs, m2 outputs, double alpha, uint epochs)
	{
		const double* inputArr = flatSamples(inputs);
		const double* outputArr = flatSamples(outputs);

		trainMlpModelRBF(this->model, inputArr, outputArr, inputs.size(), inputs[0].size(), outputs[0].size(), alpha, epochs);

		delete[] inputArr;
		delete[] outputArr;
	}
};
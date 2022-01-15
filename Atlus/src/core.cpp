#include "../headers/core.h"

#include <algorithm>
#include <cassert>
#include <time.h>
#include "../headers/utils.h"

/// <summary>
/// Fait passer les valeurs en entr�e � la couche de sortie en appliquant la tangente hyperbolique
/// de la somme pond�r�e des valeurs de sortie des neurones de chaque couche
/// </summary>
/// <param name="model">Donn�es du mod�le</param>
/// <param name="sampleInputs">Valeurs d'entr�e</param>
void forwardPassClassification(MLPData* model, const double sampleInputs[])
{
	for (uint j = 1; j < model->npl[0] + 1; ++j)
		model->X[0][j] = sampleInputs[j - 1];

	for (uint l = 1; l < model->L + 1; ++l)
	{
		for (uint j = 1; j < model->npl[l] + 1; ++j)
		{
			double total = 0.0;

			for (uint i = 0; i < model->npl[l - 1] + 1; ++i)
				total += model->W[l][i][j] * model->X[l - 1][i];

			model->X[l][j] = tanh(total);
		}
	}
}

/// <summary>
/// Fait passer les valeurs en entr�e � la couche de sortie en appliquant la tangente hyperbolique
/// de la somme pond�r�e des valeurs de sortie des neurones de chaque couche
/// La tangente hyperbolique n'est pas appliqu�e sur la couche de sortie (r�gression)
/// </summary>
/// <param name="model">Donn�es du mod�le</param>
/// <param name="sampleInputs">Valeurs d'entr�e</param>
void forwardPassRegression(MLPData* model, const double sampleInputs[])
{
	for (uint j = 1; j < model->npl[0] + 1; ++j) model->X[0][j] = sampleInputs[j - 1];

	double total;

	for (uint l = 1; l < model->L - 1; ++l)
	{
		for (uint j = 1; j < model->npl[l] + 1; ++j)
		{
			total = 0.0;

			for (uint i = 0; i < model->npl[l - 1] + 1; ++i)
				total += model->W[l][i][j] * model->X[l - 1][i];

			model->X[l][j] = tanh(total);
		}
	}

	for (uint j = 1; j < model->npl[model->L] + 1; ++j)
	{
		total = 0.0;

		for (uint i = 0; i < model->npl[model->L - 1] + 1; ++i)
			total += model->W[model->L][i][j] * model->X[model->L - 1][i];

		model->X[model->L][j] = total;
	}

	for (uint j = 1; j < model->npl[model->L] + 1; ++j)
	{
		total = 0.0;

		for (uint i = 0; i < model->npl[model->L - 1] + 1; ++i)
			total += model->W[model->L][i][j] * model->X[model->L - 1][i];

		model->X[model->L][j] = total;
	}
}

/// <summary>
/// R�tropropage les valeurs d'erreur
/// </summary>
/// <param name="model">Donn�es du mod�le</param>
/// <param name="alpha">Pas d'apprentissage</param>
void backpropagateAndLearnMlpModel(MLPData* model, const double alpha)
{
	for (uint l = model->L; l >= 2; --l)
	{
		for (uint i = 1; i < model->npl[l - 1] + 1; ++i)
		{
			double total = 0.0;

			for (uint j = 1; j < model->npl[l] + 1; ++j) total += model->W[l][i][j] * model->deltas[l][j];

			total *= (1 - model->X[l - 1][i] * model->X[l - 1][i]);
			model->deltas[l - 1][i] = total;
		}
	}

	for (uint l = 1; l < model->L + 1; ++l)
	{
		for (uint i = 0; i < model->npl[l - 1] + 1; ++i)
		{
			for (uint j = 1; j < model->npl[l] + 1; ++j)
				model->W[l][i][j] -= alpha * model->X[l - 1][i] * model->deltas[l][j];
		}
	}
}

DllExport MLPData* createMlpModel(uint npl[], uint nplSize)
{
	srand(time(NULL));

	m3 W;
	m2 X;
	m2 deltas;

	W.reserve(nplSize - 1);
	X.reserve(nplSize);
	deltas.reserve(nplSize);

	for (uint l = 0; l < nplSize; ++l)
	{
		if (l == 0)
		{
			W.push_back({});
			continue;
		}

		m2 v;

		v.reserve(npl[l - 1] + 1);

		for (uint i = 0; i < npl[l - 1] + 1; ++i)
		{
			m1 d;

			d.reserve(npl[l] + 1);

			for (uint j = 0; j < npl[l] + 1; ++j)
				d.push_back(j == 0 ? 0.0 : random() * 2.0 - 1.0);

			v.push_back(d);
		}

		W.push_back(v);
	}

	for (uint l = 0; l < nplSize; ++l)
	{
		m1 x;
		m1 d;

		x.reserve(npl[l] + 1);
		d.reserve(npl[l] + 1);

		for (uint j = 0; j < npl[l] + 1; ++j)
		{
			x.push_back(j == 0 ? 1.0 : 0.0);
			d.push_back(0.0);
		}

		X.push_back(x);
		deltas.push_back(d);
	}

	return new MLPData(W, std::vector<uint>(npl, npl + nplSize), X, deltas);
}

DllExport void trainMlpModelClassificationSingle(
	MLPData* model,
	const double sampleInputs[],
	const double sampleExpectedOutput[],
	const double alpha
)
{
	forwardPassClassification(model, sampleInputs);

	int j = 1;

	std::transform(
		model->X[model->L].begin() + 1,
		model->X[model->L].end(),
		model->deltas[model->L].begin() + 1,
		[&j, &sampleExpectedOutput](double x) -> double { return (x - sampleExpectedOutput[j++ - 1]) * (1 - x * x); }
	);

	backpropagateAndLearnMlpModel(model, alpha);
}

DllExport void trainMlpModelRegressionSingle(
	MLPData* model,
	const double sampleInputs[],
	const double sampleExpectedOutput[],
	const double alpha
)
{
	forwardPassRegression(model, sampleInputs);

	int j = 1;

	std::transform(
		model->X[model->L].begin() + 1,
		model->X[model->L].end(),
		model->deltas[model->L].begin() + 1,
		[&j, &sampleExpectedOutput](double x) -> double { return x - sampleExpectedOutput[j++ - 1]; }
	);

	backpropagateAndLearnMlpModel(model, alpha);
}

DllExport void trainMlpModelClassification(
	MLPData* model,
	const double samplesInputs[],
	const double samplesExpectedOutputs[],
	const uint sampleCount,
	const uint inputDim,
	const uint outputDim,
	const double alpha,
	uint nbIter
)
{
	assert(("Input should have as many elements as there are neurons on the first hidden layer", inputDim == model->npl[0]));
	assert(("Output should have as many elements as there are neurons on the last hidden layer", outputDim == model->npl[model->L]));

	while (nbIter--)
	{
		uint k = (rand() % sampleCount) * inputDim;

		trainMlpModelClassificationSingle(
			model,
			samplesInputs + k,
			samplesExpectedOutputs + k,
			alpha
		);
	}
}

DllExport void trainMlpModelRegression(
	MLPData* model,
	const double samplesInputs[],
	const double samplesExpectedOutputs[],
	const uint sampleCount,
	const uint inputDim,
	const uint outputDim,
	const double alpha,
	uint nbIter
)
{
	assert(("Input should have as many elements as there are neurons on the first hidden layer", inputDim == model->npl[0]));
	assert(("Output should have as many elements as there are neurons on the last hidden layer", outputDim == model->npl[model->L]));

	while (nbIter--)
	{
		uint k = (rand() % sampleCount) * inputDim;

		trainMlpModelRegressionSingle(
			model,
			samplesInputs + k,
			samplesExpectedOutputs + k,
			alpha
		);
	}
}

DllExport double* predictMlpModelClassification(
	MLPData* model,
	const double sampleInputs[]
)
{
	double* result = new double[model->X[model->L].size() - 1];

	forwardPassClassification(model, sampleInputs);
	std::copy(model->X[model->L].begin() + 1, model->X[model->L].end(), result);

	return result;
}

DllExport double* predictMlpModelRegression(
	MLPData* model,
	const double sampleInputs[]
)
{
	double* result = new double[model->X[model->L].size() - 1];

	forwardPassRegression(model, sampleInputs);
	std::copy(model->X[model->L].begin() + 1, model->X[model->L].end(), result);

	return result;
}

DllExport double evaluateModelAccuracy(
	MLPData* model,
	const double samplesInputs[],
	const double samplesExpectedOutputs[],
	const uint sampleCount,
	const uint inputDim,
	const uint outputDim
)
{
	uint totalGoodPredictions = 0;
	double* v;
	double r;

	for (uint i = 0; i < sampleCount; ++i)
	{
		v = predictMlpModelClassification(model, samplesInputs + (inputDim * i));
		r = *v;

		delete[] v;
		if (r * samplesExpectedOutputs[outputDim * i] >= 0) totalGoodPredictions += 1;
	}

	return (double)totalGoodPredictions / (double)sampleCount;
}

DllExport void destroyMlpModel(MLPData* model)
{
	delete model;
}

DllExport void destroyMlpResult(const double* result)
{
	delete[] result;
}

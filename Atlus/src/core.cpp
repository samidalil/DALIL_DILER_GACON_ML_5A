#include "headers/core.h"

#include "headers/utils.h"
#include "headers/parser.h"

#include <algorithm>
#include <cassert>
#include <time.h>
#include <iostream>

double vectorNorm(double x, double y) {
	std::complex<double> mycomplex(x, y);
	return (sqrt(std::norm(mycomplex)));
}

void displayModelData(MLPData* model)
{
	std::cout << "Valeurs des poids entre les neurones" << std::endl;
	
	for (uint l = 1; l <= model->L; ++l)
	{
		std::cout << "Couche " << l << " :" << std::endl;

		for (uint i = 0; i < model->npl[l - 1] + 1; ++i)
		{
			std::cout << "- avec couche " << i << " :" << std::endl;

			for (uint j = 0; j < model->npl[l] + 1; ++j)
				std::cout << model->W[l][i][j] << ", ";

			std::cout << std::endl;
		}
	}

	std::cout << std::endl << "Valeurs des sorties" << std::endl;

	for (uint l = 0; l <= model->L; ++l)
	{
		std::cout << "Couche " << l + 1 << " :" << std::endl;

		for (uint j = 0; j <= model->npl[l]; ++j)
			std::cout << model->X[l][j] << ", ";
		
		std::cout << std::endl;
	}
	
	std::cout << std::endl << "Valeurs des différences" << std::endl;

	for (uint l = 0; l <= model->L; ++l)
	{
		std::cout << "Couche " << l + 1 << " :" << std::endl;

		for (uint j = 0; j <= model->npl[l]; ++j)
			std::cout << model->deltas[l][j] << ", ";

		std::cout << std::endl;
	}
}

#define DEBUG 0

#if DEBUG
#	define DEBUG_displayModelData(model) { displayModelData(model); }
#else
#	define DEBUG_displayModelData(model) {}
#endif

constexpr double ERROR_EPSILON = 0.01;
constexpr double GAMMA_RBF = 0.2;

/// <summary>
/// Fait passer les valeurs en entrée à la couche de sortie en appliquant la tangente hyperbolique
/// de la somme pondérée des valeurs de sortie des neurones de chaque couche
/// </summary>
/// <param name="model">Adresse du modèle</param>
/// <param name="sampleInputs">Valeurs d'entrée</param>
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
/// Fait passer les valeurs en entrée à la couche de sortie en appliquant la tangente hyperbolique
/// de la somme pondérée des valeurs de sortie des neurones de chaque couche
/// La tangente hyperbolique n'est pas appliquée sur la couche de sortie (régression)
/// </summary>
/// <param name="model">Adresse du modèle</param>
/// <param name="sampleInputs">Valeurs d'entrée</param>
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
///
/// </summary>
/// <param name="model">Adresse du modèle</param>
/// <param name="sampleInputs">Valeurs d'entrée</param>
void forwardPassRBF(MLPData* model, const double sampleInputs[])
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

			model->X[l][j] = tanh( total*exp(-GAMMA_RBF * pow(vectorNorm(model->X[l][j], model->W[l][model->npl[l - 1]][j]), 2)) );
			//model->X[l][j] = tanh(total);
		}
	}

	for (uint j = 1; j < model->npl[model->L] + 1; ++j)
	{
		total = 0.0;

		for (uint i = 0; i < model->npl[model->L - 1] + 1; ++i)
			total += model->W[model->L][i][j] * model->X[model->L - 1][i];

		//model->X[model->L][j] = total * exp(-GAMMA_RBF * pow(vectorNorm(model->X[model->L][j], model->W[model->L][model->npl[model->L - 1]][j]), 2));
		model->X[model->L][j] = total;
	}

	for (uint j = 1; j < model->npl[model->L] + 1; ++j)
	{
		total = 0.0;

		for (uint i = 0; i < model->npl[model->L - 1] + 1; ++i)
			total += model->W[model->L][i][j] * model->X[model->L - 1][i];

		//model->X[model->L][j] = total * exp(-GAMMA_RBF * pow(vectorNorm(model->X[model->L][j], model->W[model->L][model->npl[model->L - 1]][j]), 2));
		model->X[model->L][j] = total;
	}
}

/// <summary>
/// Rétropropage les valeurs d'erreur
/// </summary>
/// <param name="model">Adresse du modèle</param>
/// <param name="alpha">Pas d'apprentissage</param>
void backpropagateAndLearnMlpModel(MLPData* model, const double alpha)
{
	for (int l = model->L; l > 1; --l)
	{
		for (int i = 1; i < model->npl[l - 1] + 1; ++i)
		{
			double total = 0.0;
			
			for (int j = 1; j < model->npl[l] + 1; ++j)
				total += model->W[l][i][j] * model->deltas[l][j];

			total *= (1 - model->X[l - 1][i] * model->X[l - 1][i]);
			model->deltas[l - 1][i] = total;
		}
	}

	for (int l = 1; l < model->L + 1; ++l)
	{
		for (int i = 0; i < model->npl[l - 1] + 1; ++i)
		{
			for (int j = 1; j < model->npl[l] + 1; ++j)
				model->W[l][i][j] -= alpha * model->X[l - 1][i] * model->deltas[l][j];
		}
	}
}

/// <summary>
/// Retourne un pointeur vers les données d'un modèle nouvellement généré
/// </summary>
/// <param name="npl">Tableau contenant le nombre de neurones par couche</param>
/// <param name="nplSize">Taille du tableau</param>
/// <returns>L'adresse du modèle</returns>
DllExport MLPData* createMlpModel(uint npl[], uint nplSize)
{
	srand(time(NULL));

	m3 W;
	m2 X;
	m2 deltas;

	W.reserve(nplSize);
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

/// <summary>
/// Crée une structure de données pour le modèle à partir de la chaine de caractères fournie
/// </summary>
/// <param name="data">Chaine de caractères contenant les informations de nombres de neurones par couches et de poids entre les neurones</param>
/// <returns>L'adresse du modèle initialisé</returns>
DllExport MLPData* deserializeModel(std::string data)
{
	return deserialize(data);
}

/// <summary>
/// Transforme le contenu du modèle en une chaine de caractères
/// </summary>
/// <param name="model">Adresse du modèle</param>
/// <returns>Une chaine de caractères contenant les données du modèle</returns>
DllExport std::string serializeModel(MLPData* model)
{
	return serialize(model);
}

/// <summary>
/// Entraîne le modèle pour de la classification avec une entrée et sa sortie correspondante
/// </summary>
/// <param name="model">Adresse du modèle</param>
/// <param name="sampleInputs">Valeurs d'entrée</param>
/// <param name="sampleExpectedOutput">Valeurs de sortie</param>
/// <param name="alpha">Pas d'apprentissage</param>
DllExport void trainMlpModelClassificationSingle(
	MLPData* model,
	const double sampleInputs[],
	const double sampleExpectedOutput[],
	const double alpha
)
{
	forwardPassClassification(model, sampleInputs);

	for (int j = 1; j < model->npl[model->L] + 1; ++j)
	{
		double x = model->X[model->L][j];
		model->deltas[model->L][j] = (x - sampleExpectedOutput[j - 1]) * (1 - x * x);
	}

	backpropagateAndLearnMlpModel(model, alpha);
}

/// <summary>
/// Entraîne le modèle pour de la régression avec une entrée et sa sortie correspondante
/// </summary>
/// <param name="model">Adresse du modèle</param>
/// <param name="sampleInputs">Valeurs d'entrée</param>
/// <param name="sampleExpectedOutput">Valeurs de sortie</param>
/// <param name="alpha">Pas d'apprentissage</param>
DllExport void trainMlpModelRegressionSingle(
	MLPData* model,
	const double sampleInputs[],
	const double sampleExpectedOutput[],
	const double alpha
)
{
	forwardPassRegression(model, sampleInputs);

	for (int j = 1; j < model->npl[model->L] + 1; ++j)
		model->deltas[model->L][j] = model->X[model->L][j] - sampleExpectedOutput[j - 1];

	backpropagateAndLearnMlpModel(model, alpha);
}

/// <summary>
/// Entraîne le modèle pour de la RBF Naïf avec une entrée et sa sortie correspondante
/// </summary>
/// <param name="model">Adresse du modèle</param>
/// <param name="sampleInputs">Valeurs d'entrée</param>
/// <param name="sampleExpectedOutput">Valeurs de sortie</param>
/// <param name="alpha">Pas d'apprentissage</param>
DllExport void trainMlpModelRBFSingle(
	MLPData* model,
	const double sampleInputs[],
	const double sampleExpectedOutput[],
	const double alpha
)
{
	forwardPassRBF(model, sampleInputs);

	for (int j = 1; j < model->npl[model->L] + 1; ++j)
		model->deltas[model->L][j] = model->X[model->L][j] - sampleExpectedOutput[j - 1];

	backpropagateAndLearnMlpModel(model, alpha);
}

/// <summary>
/// Entraîne le modèle pour de la classification avec plusieurs entrées et leurs sorties correspondantes
/// </summary>
/// <param name="model">Adresse du modèle</param>
/// <param name="samplesInputs">Tableau d'entrées</param>
/// <param name="samplesExpectedOutputs">Tableau de sorties</param>
/// <param name="sampleCount">Nombre de samples d'entraînement</param>
/// <param name="inputDim">Taille d'une entrée</param>
/// <param name="outputDim">Taille d'une sortie</param>
/// <param name="alpha">Pas d'apprentissage</param>
/// <param name="epochs">Nombre d'epochs pour cet entraînement</param>
DllExport void trainMlpModelClassification(
	MLPData* model,
	const double samplesInputs[],
	const double samplesExpectedOutputs[],
	const uint sampleCount,
	const uint inputDim,
	const uint outputDim,
	const double alpha,
	uint epochs
)
{
	assert(("Input should have as many elements as there are neurons on the first hidden layer", inputDim == model->npl[0]));
	assert(("Output should have as many elements as there are neurons on the last hidden layer", outputDim == model->npl[model->L]));

	while (epochs--)
	{
		DEBUG_displayModelData(model);

		uint randomSampleIndex = (rand() % sampleCount);

		trainMlpModelClassificationSingle(
			model,
			samplesInputs + (randomSampleIndex * inputDim),
			samplesExpectedOutputs + (randomSampleIndex * outputDim),
			alpha
		);

	}

	DEBUG_displayModelData(model);
}

/// <summary>
/// Entraîne le modèle pour de la régression avec plusieurs entrées et leurs sorties correspondantes
/// </summary>
/// <param name="model">Adresse du modèle</param>
/// <param name="samplesInputs">Tableau d'entrées</param>
/// <param name="samplesExpectedOutputs">Tableau de sorties</param>
/// <param name="sampleCount">Nombre d'échantillons d'entraînement</param>
/// <param name="inputDim">Taille d'une entrée</param>
/// <param name="outputDim">Taille d'une sortie</param>
/// <param name="alpha">Pas d'apprentissage</param>
/// <param name="epochs">Nombre d'epochs pour cet entraînement</param>
DllExport void trainMlpModelRegression(
	MLPData* model,
	const double samplesInputs[],
	const double samplesExpectedOutputs[],
	const uint sampleCount,
	const uint inputDim,
	const uint outputDim,
	const double alpha,
	uint epochs
)
{
	assert(("Input should have as many elements as there are neurons on the first hidden layer", inputDim == model->npl[0]));
	assert(("Output should have as many elements as there are neurons on the last hidden layer", outputDim == model->npl[model->L]));

	while (epochs--)
	{
		DEBUG_displayModelData(model);
		
		uint randomSampleIndex = rand() % sampleCount;

		trainMlpModelRegressionSingle(
			model,
			samplesInputs + (randomSampleIndex * inputDim),
			samplesExpectedOutputs + (randomSampleIndex * outputDim),
			alpha
		);
	}

	DEBUG_displayModelData(model);
}

/// <summary>
/// Entraîne le modèle pour de la rbf naïf avec plusieurs entrées et leurs sorties correspondantes
/// </summary>
/// <param name="model">Adresse du modèle</param>
/// <param name="samplesInputs">Tableau d'entrées</param>
/// <param name="samplesExpectedOutputs">Tableau de sorties</param>
/// <param name="sampleCount">Nombre d'échantillons d'entraînement</param>
/// <param name="inputDim">Taille d'une entrée</param>
/// <param name="outputDim">Taille d'une sortie</param>
/// <param name="alpha">Pas d'apprentissage</param>
/// <param name="epochs">Nombre d'epochs pour cet entraînement</param>
DllExport void trainMlpModelRBF(
	MLPData* model,
	const double samplesInputs[],
	const double samplesExpectedOutputs[],
	const uint sampleCount,
	const uint inputDim,
	const uint outputDim,
	const double alpha,
	uint epochs
)
{
	assert(("Input should have as many elements as there are neurons on the first hidden layer", inputDim == model->npl[0]));
	assert(("Output should have as many elements as there are neurons on the last hidden layer", outputDim == model->npl[model->L]));

	while (epochs--)
	{
		DEBUG_displayModelData(model);

		uint randomSampleIndex = rand() % sampleCount;

		trainMlpModelRBFSingle(
			model,
			samplesInputs + (randomSampleIndex * inputDim),
			samplesExpectedOutputs + (randomSampleIndex * outputDim),
			alpha
		);
	}

	DEBUG_displayModelData(model);
}

/// <summary>
/// Retourne la prédiction de classification pour l'entrée fournie
/// </summary>
/// <param name="model">Adresse du modèle</param>
/// <param name="sampleInputs">Valeurs d'entrée</param>
/// <returns>Valeurs de sorties prédites</returns>
DllExport double* predictMlpModelClassification(
	MLPData* model,
	const double sampleInputs[]
)
{
	double* result = new double[model->npl[model->L]];

	forwardPassClassification(model, sampleInputs);
	std::copy(model->X[model->L].begin() + 1, model->X[model->L].end(), result);

	return result;
}

/// <summary>
/// Retourne la prédiction de régression pour l'entrée fournie
/// </summary>
/// <param name="model">Adresse du modèle</param>
/// <param name="sampleInputs">Valeurs d'entrée</param>
/// <returns>Valeurs de sorties prédites</returns>
DllExport double* predictMlpModelRegression(
	MLPData* model,
	const double sampleInputs[]
)
{
	double* result = new double[model->npl[model->L]];

	forwardPassRegression(model, sampleInputs);
	std::copy(model->X[model->L].begin() + 1, model->X[model->L].end(), result);

	return result;
}

/// <summary>
/// Retourne la prédiction de rbf naïf pour l'entrée fournie
/// </summary>
/// <param name="model">Adresse du modèle</param>
/// <param name="sampleInputs">Valeurs d'entrée</param>
/// <returns>Valeurs de sorties prédites</returns>
DllExport double* predictMlpModelRBF(
	MLPData* model,
	const double sampleInputs[]
)
{
	double* result = new double[model->npl[model->L]];

	forwardPassRBF(model, sampleInputs);
	std::copy(model->X[model->L].begin() + 1, model->X[model->L].end(), result);

	return result;
}

/// <summary>
/// Evalue le taux de précision du modèle de classification
/// </summary>
/// <param name="model">Adresse du modèle</param>
/// <param name="samplesInputs">Tableau d'entrées</param>
/// <param name="samplesExpectedOutputs">Tableau de sorties</param>
/// <param name="sampleCount">Nombre d'échantillons de test</param>
/// <param name="inputDim">Taille d'une entrée</param>
/// <param name="outputDim">Taille d'une sortie</param>
/// <returns>Taux de précision du modèle</returns>
DllExport double evaluateModelAccuracyClassification(
	MLPData* model,
	const double samplesInputs[],
	const double samplesExpectedOutputs[],
	const uint sampleCount,
	const uint inputDim,
	const uint outputDim
)
{
	uint totalGoodPredictions = 0;
	bool good;
	double* prediction;

	for (uint i = 0; i < sampleCount; ++i)
	{
		const double* input = samplesInputs + (inputDim * i);
		const double* output = samplesExpectedOutputs + (outputDim * i);

		prediction = predictMlpModelClassification(model, input);
		good = true;

		for (uint j = 0; j < outputDim; ++j)
		{
			if (prediction[j] * output[j] < 0)
				good = false;
		}

		delete[] prediction;
	
		if (good) ++totalGoodPredictions;

	}

	return (double)totalGoodPredictions / (double)sampleCount;
}

/// <summary>
/// Evalue le taux de précision du modèle de régression
/// </summary>
/// <param name="model">Adresse du modèle</param>
/// <param name="samplesInputs">Tableau d'entrées</param>
/// <param name="samplesExpectedOutputs">Tableau de sorties</param>
/// <param name="sampleCount">Nombre d'échantillons de test</param>
/// <param name="inputDim">Taille d'une entrée</param>
/// <param name="outputDim">Taille d'une sortie</param>
/// <returns>Taux de précision du modèle</returns>
DllExport double evaluateModelAccuracyRegression(
	MLPData* model,
	const double samplesInputs[],
	const double samplesExpectedOutputs[],
	const uint sampleCount,
	const uint inputDim,
	const uint outputDim
)
{
	uint totalGoodPredictions = 0;
	bool good;
	double* prediction;

	for (uint i = 0; i < sampleCount; ++i)
	{
		const double* input = samplesInputs + (inputDim * i);
		const double* output = samplesExpectedOutputs + (outputDim * i);

		prediction = predictMlpModelRegression(model, input);
		good = true;

		for (uint j = 0; j < outputDim; ++j)
		{
			if (abs(prediction[j] - output[j]) > ERROR_EPSILON)
				good = false;
		}

		delete[] prediction;

		if (good) ++totalGoodPredictions;

	}

	return (double)totalGoodPredictions / (double)sampleCount;
}

/// <summary>
/// Evalue le taux de précision du modèle de régression
/// </summary>
/// <param name="model">Adresse du modèle</param>
/// <param name="samplesInputs">Tableau d'entrées</param>
/// <param name="samplesExpectedOutputs">Tableau de sorties</param>
/// <param name="sampleCount">Nombre d'échantillons de test</param>
/// <param name="inputDim">Taille d'une entrée</param>
/// <param name="outputDim">Taille d'une sortie</param>
/// <returns>Taux de précision du modèle</returns>
DllExport double evaluateModelAccuracyRBF(
	MLPData* model,
	const double samplesInputs[],
	const double samplesExpectedOutputs[],
	const uint sampleCount,
	const uint inputDim,
	const uint outputDim
)
{
	uint totalGoodPredictions = 0;
	bool good;
	double* prediction;

	for (uint i = 0; i < sampleCount; ++i)
	{
		const double* input = samplesInputs + (inputDim * i);
		const double* output = samplesExpectedOutputs + (outputDim * i);

		prediction = predictMlpModelRBF(model, input);
		good = true;

		for (uint j = 0; j < outputDim; ++j)
		{
			if (abs(prediction[j] - output[j]) > ERROR_EPSILON)
				good = false;
		}

		delete[] prediction;

		if (good) ++totalGoodPredictions;

	}

	return (double)totalGoodPredictions / (double)sampleCount;
}

/// <summary>
/// Détruit la ressource du modèle
/// </summary>
/// <param name="model">Adresse du modèle</param>
DllExport void destroyMlpModel(MLPData* model)
{
	delete model;
}

/// <summary>
/// Détruit un tableau de résultat de prédiction du modèle
/// </summary>
/// <param name="result">Adresse du tableau de données</param>
/// <returns></returns>
DllExport void destroyMlpResult(const double* result)
{
	delete[] result;
}

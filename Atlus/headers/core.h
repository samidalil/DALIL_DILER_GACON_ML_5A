#pragma once

#include "MLPData.h"

#ifndef DllExport
#define DllExport extern "C" __declspec(dllexport)
#endif

// TD: Passer les fonctions en m�thodes de classe et rajouter des versions imp�ratives pour les exports ?
// TD: Commenter le reste des fonctions

/// <summary>
/// Fait passer les valeurs en entr�e � la couche de sortie en appliquant la tangente hyperbolique
/// de la somme pond�r�e des valeurs de sortie des neurones de chaque couche
/// </summary>
/// <param name="model">Donn�es du mod�le</param>
/// <param name="sampleInputs">Valeurs d'entr�e</param>
void forwardPassClassification(MLPData* model, const double sampleInputs[]);

/// <summary>
/// Fait passer les valeurs en entr�e � la couche de sortie en appliquant la tangente hyperbolique
/// de la somme pond�r�e des valeurs de sortie des neurones de chaque couche
/// La tangente hyperbolique n'est pas appliqu�e sur la couche de sortie (r�gression)
/// </summary>
/// <param name="model">Donn�es du mod�le</param>
/// <param name="sampleInputs">Valeurs d'entr�e</param>
void forwardPassRegression(MLPData* model, const double sampleInputs[]);

/// <summary>
/// R�tropropage les valeurs d'erreur
/// </summary>
/// <param name="model">Donn�es du mod�le</param>
/// <param name="alpha">Pas d'apprentissage</param>
void backpropagateAndLearnMlpModel(MLPData* model, const double alpha);

DllExport MLPData* createMlpModel(uint npl[], uint nplSize);

DllExport void trainMlpModelClassificationSingle(
	MLPData* model,
	const double sampleInputs[],
	const double sampleExpectedOutput[],
	const double alpha
);

DllExport void trainMlpModelRegressionSingle(
	MLPData* model,
	const double sampleInputs[],
	const double sampleExpectedOutput[],
	const double alpha
);

DllExport void trainMlpModelClassification(
	MLPData* model,
	const double samplesInputs[],
	const double samplesExpectedOutputs[],
	const uint sampleCount,
	const uint inputDim,
	const uint outputDim,
	const double alpha,
	uint nbIter
);

DllExport void trainMlpModelRegression(
	MLPData* model,
	const double samplesInputs[],
	const double samplesExpectedOutputs[],
	const uint sampleCount,
	const uint inputDim,
	const uint outputDim,
	const double alpha,
	uint nbIter
);

DllExport double* predictMlpModelClassification(MLPData* model, const double sampleInputs[]);

DllExport double* predictMlpModelRegression(MLPData* model, const double sampleInputs[]);

DllExport double evaluateModelAccuracy(
	MLPData* model,
	const double samplesInputs[],
	const double samplesExpectedOutputs[],
	const uint sampleCount,
	const uint inputDim,
	const uint outputDim
);

DllExport void destroyMlpModel(MLPData* model);

DllExport void destroyMlpResult(const double* result);

#pragma once

#include "MLPData.h"

#ifndef DllExport
#define DllExport extern "C" __declspec(dllexport)
#endif

#pragma region Core Functions

// TD: Passer les fonctions en méthodes de classe et rajouter des versions impératives pour les exports ?
// TD: Commenter le reste des fonctions

/// <summary>
/// Fait passer les valeurs en entrée à la couche de sortie en appliquant la tangente hyperbolique
/// de la somme pondérée des valeurs de sortie des neurones de chaque couche
/// </summary>
/// <param name="model">Données du modèle</param>
/// <param name="sampleInputs">Valeurs d'entrée</param>
void forwardPassClassification(MLPData* model, const double sampleInputs[]);

/// <summary>
/// Fait passer les valeurs en entrée à la couche de sortie en appliquant la tangente hyperbolique
/// de la somme pondérée des valeurs de sortie des neurones de chaque couche
/// La tangente hyperbolique n'est pas appliquée sur la couche de sortie (régression)
/// </summary>
/// <param name="model">Données du modèle</param>
/// <param name="sampleInputs">Valeurs d'entrée</param>
void forwardPassRegression(MLPData* model, const double sampleInputs[]);

/// <summary>
/// Rétropropage les valeurs d'erreur
/// </summary>
/// <param name="model">Données du modèle</param>
/// <param name="alpha">Pas d'apprentissage</param>
void backpropagateAndLearnMlpModel(MLPData* model, const double alpha);

#pragma endregion

#pragma region Exported Methods

/// <summary>
/// Retourne un pointeur vers les données d'un modèle nouvellement généré
/// </summary>
/// <param name="npl">Tableau contenant le nombre de neurones par couche</param>
/// <param name="nplSize">Taille du tableau</param>
/// <returns>L'adresse du modèle</returns>
DllExport MLPData* createMlpModel(uint npl[], uint nplSize);

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
);

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
);

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
	uint nbIter
);

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
	uint nbIter
);

/// <summary>
/// Retourne la prédiction de classification pour l'entrée fournie
/// </summary>
/// <param name="model">Adresse du modèle</param>
/// <param name="sampleInputs">Valeurs d'entrée</param>
/// <returns>Valeurs de sorties prédites</returns>
DllExport double* predictMlpModelClassification(MLPData* model, const double sampleInputs[]);

/// <summary>
/// Retourne la prédiction de régression pour l'entrée fournie
/// </summary>
/// <param name="model">Adresse du modèle</param>
/// <param name="sampleInputs">Valeurs d'entrée</param>
/// <returns>Valeurs de sorties prédites</returns>
DllExport double* predictMlpModelRegression(MLPData* model, const double sampleInputs[]);

/// <summary>
/// Evalue le taux de précision du modèle
/// </summary>
/// <param name="model">Adresse du modèle</param>
/// <param name="samplesInputs">Tableau d'entrées</param>
/// <param name="samplesExpectedOutputs">Tableau de sorties</param>
/// <param name="sampleCount">Nombre d'échantillons de test</param>
/// <param name="inputDim">Taille d'une entrée</param>
/// <param name="outputDim">Taille d'une sortie</param>
/// <returns>Taux de précision du modèle</returns>
DllExport double evaluateModelAccuracy(
	MLPData* model,
	const double samplesInputs[],
	const double samplesExpectedOutputs[],
	const uint sampleCount,
	const uint inputDim,
	const uint outputDim
);

/// <summary>
/// Détruit la ressource du modèle
/// </summary>
/// <param name="model">Adresse du modèle</param>
DllExport void destroyMlpModel(MLPData* model);

/// <summary>
/// Détruit un tableau de résultat de prédiction du modèle
/// </summary>
/// <param name="result">Adresse du tableau de données</param>
/// <returns></returns>
DllExport void destroyMlpResult(const double* result);

#pragma endregion

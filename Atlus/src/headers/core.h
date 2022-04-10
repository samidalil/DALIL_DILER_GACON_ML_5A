#pragma once

#include "MLPData.h"

#include <string>

#ifndef DllExport
#define DllExport extern "C" __declspec(dllexport)
#endif

#pragma region Core Functions

// TD: Passer les fonctions en m�thodes de classe et rajouter des versions imp�ratives pour les exports ?

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

#pragma endregion

#pragma region Exported Methods

/// <summary>
/// Retourne un pointeur vers les donn�es d'un mod�le nouvellement g�n�r�
/// </summary>
/// <param name="npl">Tableau contenant le nombre de neurones par couche</param>
/// <param name="nplSize">Taille du tableau</param>
/// <returns>L'adresse du mod�le</returns>
DllExport MLPData* createMlpModel(uint npl[], uint nplSize);

/// <summary>
/// Cr�e une structure de donn�es pour le mod�le � partir de la chaine de caract�res fournie
/// </summary>
/// <param name="data">Chaine de caract�res contenant les informations de nombres de neurones par couches et de poids entre les neurones</param>
/// <returns>L'adresse du mod�le initialis�</returns>
DllExport MLPData* deserializeModel(const char* data);

/// <summary>
/// Transforme le contenu du mod�le en une chaine de caract�res
/// </summary>
/// <param name="model">Adresse du mod�le</param>
/// <returns>Une chaine de caract�res contenant les donn�es du mod�le</returns>
DllExport char* serializeModel(MLPData* model);

/// <summary>
/// Entra�ne le mod�le pour de la classification avec une entr�e et sa sortie correspondante
/// </summary>
/// <param name="model">Adresse du mod�le</param>
/// <param name="sampleInputs">Valeurs d'entr�e</param>
/// <param name="sampleExpectedOutput">Valeurs de sortie</param>
/// <param name="alpha">Pas d'apprentissage</param>
DllExport void trainMlpModelClassificationSingle(
	MLPData* model,
	const double sampleInputs[],
	const double sampleExpectedOutput[],
	const double alpha
);

/// <summary>
/// Entra�ne le mod�le pour de la r�gression avec une entr�e et sa sortie correspondante
/// </summary>
/// <param name="model">Adresse du mod�le</param>
/// <param name="sampleInputs">Valeurs d'entr�e</param>
/// <param name="sampleExpectedOutput">Valeurs de sortie</param>
/// <param name="alpha">Pas d'apprentissage</param>
DllExport void trainMlpModelRegressionSingle(
	MLPData* model,
	const double sampleInputs[],
	const double sampleExpectedOutput[],
	const double alpha
);

/// <summary>
/// Entra�ne le mod�le pour de la classification avec plusieurs entr�es et leurs sorties correspondantes
/// </summary>
/// <param name="model">Adresse du mod�le</param>
/// <param name="samplesInputs">Tableau d'entr�es</param>
/// <param name="samplesExpectedOutputs">Tableau de sorties</param>
/// <param name="sampleCount">Nombre de samples d'entra�nement</param>
/// <param name="inputDim">Taille d'une entr�e</param>
/// <param name="outputDim">Taille d'une sortie</param>
/// <param name="alpha">Pas d'apprentissage</param>
/// <param name="epochs">Nombre d'epochs pour cet entra�nement</param>
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
/// Entra�ne le mod�le pour de la r�gression avec plusieurs entr�es et leurs sorties correspondantes
/// </summary>
/// <param name="model">Adresse du mod�le</param>
/// <param name="samplesInputs">Tableau d'entr�es</param>
/// <param name="samplesExpectedOutputs">Tableau de sorties</param>
/// <param name="sampleCount">Nombre d'�chantillons d'entra�nement</param>
/// <param name="inputDim">Taille d'une entr�e</param>
/// <param name="outputDim">Taille d'une sortie</param>
/// <param name="alpha">Pas d'apprentissage</param>
/// <param name="epochs">Nombre d'epochs pour cet entra�nement</param>
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
/// Retourne la pr�diction de classification pour l'entr�e fournie
/// </summary>
/// <param name="model">Adresse du mod�le</param>
/// <param name="sampleInputs">Valeurs d'entr�e</param>
/// <returns>Valeurs de sorties pr�dites</returns>
DllExport double* predictMlpModelClassification(MLPData* model, const double sampleInputs[]);

/// <summary>
/// Retourne la pr�diction de r�gression pour l'entr�e fournie
/// </summary>
/// <param name="model">Adresse du mod�le</param>
/// <param name="sampleInputs">Valeurs d'entr�e</param>
/// <returns>Valeurs de sorties pr�dites</returns>
DllExport double* predictMlpModelRegression(MLPData* model, const double sampleInputs[]);

/// <summary>
/// Evalue le taux de pr�cision du mod�le de classification
/// </summary>
/// <param name="model">Adresse du mod�le</param>
/// <param name="samplesInputs">Tableau d'entr�es</param>
/// <param name="samplesExpectedOutputs">Tableau de sorties</param>
/// <param name="sampleCount">Nombre d'�chantillons de test</param>
/// <param name="inputDim">Taille d'une entr�e</param>
/// <param name="outputDim">Taille d'une sortie</param>
/// <returns>Taux de pr�cision du mod�le</returns>
DllExport double evaluateModelAccuracyClassification(
	MLPData* model,
	const double samplesInputs[],
	const double samplesExpectedOutputs[],
	const uint sampleCount,
	const uint inputDim,
	const uint outputDim
);

/// <summary>
/// Evalue le taux de pr�cision du mod�le de r�gression
/// </summary>
/// <param name="model">Adresse du mod�le</param>
/// <param name="samplesInputs">Tableau d'entr�es</param>
/// <param name="samplesExpectedOutputs">Tableau de sorties</param>
/// <param name="sampleCount">Nombre d'�chantillons de test</param>
/// <param name="inputDim">Taille d'une entr�e</param>
/// <param name="outputDim">Taille d'une sortie</param>
/// <returns>Taux de pr�cision du mod�le</returns>
DllExport double evaluateModelAccuracyRegression(
	MLPData* model,
	const double samplesInputs[],
	const double samplesExpectedOutputs[],
	const uint sampleCount,
	const uint inputDim,
	const uint outputDim
);

/// <summary>
/// D�truit la ressource du mod�le
/// </summary>
/// <param name="model">Adresse du mod�le</param>
DllExport void destroyMlpModel(MLPData* model);

/// <summary>
/// D�truit un tableau de r�sultat de pr�diction du mod�le
/// </summary>
/// <param name="result">Adresse du tableau de donn�es</param>
/// <returns></returns>
DllExport void destroyMlpResult(const double* result);

/// <summary>
/// D�truit la ch��ne de caract�res de s�rialisation du mod�le
/// </summary>
/// <param name="data">Donn�es s�rialis�es du mod�le</param>
/// <returns></returns>
DllExport void destroyMlpSerializedData(const char* data);

#pragma endregion

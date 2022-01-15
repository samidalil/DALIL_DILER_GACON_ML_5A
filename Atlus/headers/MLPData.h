#pragma once

#include <vector>

using m1 = std::vector<double>;
using m2 = std::vector<m1>;
using m3 = std::vector<m2>;
using uint = unsigned int;

/// <summary>
/// Représente la structure de données d'un modèle de prédiction
/// </summary>
struct MLPData
{
	// Matrice de différences de sortie des neurones avec la sortie attendue
	m2 deltas;

	// Indice de la dernière couche de neurones
	uint L;

	// Nombre de neurones par couche
	std::vector<uint> npl;

	// Matrice de valeurs de sortie de chaque neurone
	m2 X;

	// Matrice de poids de passage entre neurones
	m3 W;

	/// <summary>
	/// Constructeur de la structure de données
	/// </summary>
	/// <param name="W">Matrice de poids</param>
	/// <param name="npl">Neurones par couche</param>
	/// <param name="X">Matrice de sorties</param>
	/// <param name="deltas">Matrice de différences</param>
	MLPData(m3 W, std::vector<uint> npl, m2 X, m2 deltas) : W(W), npl(npl), X(X), deltas(deltas), L(npl.size() - 1) {}
};

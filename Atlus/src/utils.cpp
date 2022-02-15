#include "headers/utils.h"

#include <stdlib.h>

constexpr double RAND_MAX_D = (double)RAND_MAX + 1.0;

/// <summary>
/// Génère un nombre pseudo-aléatoire
/// </summary>
/// <returns>Un nombre décimal aléatoire entre 0 inclus et 1 non inclus</returns>
double random()
{
	return (double)rand() / RAND_MAX_D;
}

std::vector<std::string> split(const std::string& str, const char separator)
{
	std::string copy = str;
	int pos = 0;
	std::vector<std::string> vec;

	while ((pos = copy.find(separator)) != std::string::npos)
	{
		vec.push_back(copy.substr(0, pos));
		copy.erase(0, pos + 1);
	}

	vec.push_back(copy);

	return vec;
}
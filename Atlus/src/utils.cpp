#include "headers/utils.h"

#include <stdlib.h>

constexpr double RAND_MAX_D = (double)RAND_MAX + (double)1;

/// <summary>
/// G�n�re un nombre pseudo-al�atoire
/// </summary>
/// <returns>Un nombre d�cimal al�atoire entre 0 inclus et 1 non inclus</returns>
double random()
{
	return (double)rand() / RAND_MAX_D;
}

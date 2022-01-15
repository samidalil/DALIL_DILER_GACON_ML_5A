#include "../headers/utils.h"

#include <stdlib.h>

double random()
{
	// TD: Doit retourner une valeur entre 0 inclus et 1 non inclus, pas 0 inclus et 1 inclus
	return (double)rand() / RAND_MAX;
}

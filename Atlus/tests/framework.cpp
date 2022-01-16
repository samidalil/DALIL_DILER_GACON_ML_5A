#include "headers/framework.h"

#include <iostream>

std::queue<TestMethodCollector*> TestMethodCollector::instances;

int main()
{
	clock_t start;
	float delay;
	TestMethodCollector* i;
	
	while (!TestMethodCollector::instances.empty())
	{
		i = TestMethodCollector::instances.front();
		
		std::cout << "Beginning test \"" << i->getName() << "\" :" << std::endl << std::endl;
		
		start = clock();
		i->invoke();
		delay = ((float)clock() - start) / CLOCKS_PER_SEC;
		
		std::cout << std::endl << "Finished test \"" << i->getName() << "\" in " << delay << " seconds" << std::endl;

		TestMethodCollector::instances.pop();
	}
	return 0;
}

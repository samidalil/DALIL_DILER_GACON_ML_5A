#include "headers/framework.h"

#include <stack>
#include <iostream>

std::stack<TestMethod*> TestMethod::instances;

int main()
{
	clock_t start;
	clock_t end;
	TestMethod* i;
	
	while (!TestMethod::instances.empty())
	{
		i = TestMethod::instances.top();
		
		std::cout << "Beginning test \"" << i->getName() << "\" :" << std::endl << std::endl;
		
		start = clock();
		i->invoke();
		end = clock();
		
		std::cout << std::endl << "Finished test \"" << i->getName() << "\" in " << ((float)end - start) / CLOCKS_PER_SEC << " seconds" << std::endl;

		TestMethod::instances.pop();
	}
	return 0;
}

#include "headers/framework.h"

#include <iostream>
#include <exception>

std::queue<TestMethodCollector*> TestMethodCollector::instances;

int main()
{
	clock_t start;
	float delay;
	TestMethodCollector* i;
	bool isNotEmpty = !TestMethodCollector::instances.empty();
	
	while (isNotEmpty)
	{
		i = TestMethodCollector::instances.front();
		
		std::cout << "Beginning test \"" << i->getName() << "\" :" << std::endl << std::endl;
		
		try
		{
			start = clock();
			i->invoke();
			delay = ((float)clock() - start) / CLOCKS_PER_SEC;


			std::cout << std::endl << "Finished test \"" << i->getName() << "\" in " << delay << " seconds" << std::endl;
		}
		catch (...)
		{
			const std::exception_ptr e = std::current_exception();

			try
			{
				if (e) std::rethrow_exception(e);
			}
			catch (const std::exception& e)
			{
				std::cerr << std::endl << "Error during test \"" << i->getName() << "\" : " << e.what() << std::endl;
			}
		}

		TestMethodCollector::instances.pop();

		if ((isNotEmpty = !TestMethodCollector::instances.empty()))
			std::cout << std::endl;
	}
	return 0;
}

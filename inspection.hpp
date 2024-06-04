#pragma once
#include <vector>

namespace inspection
{
	template<typename T>
	void print_vector_element(const std::vector<T>& input)
	{
		std::cout << "inspecting vector: " << endl;
		for (auto i : input)
		{
			
			std::cout << i << " ";
		}

		std::cout << endl;

		return ;
	}
}
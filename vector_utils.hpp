#pragma once
#include <vector>

using namespace std;

namespace vector_opt
{
	size_t element_product(vector<int> input)
	{
		size_t product = 0;
		size_t layer_size = input.size();

		for (int j = 0; j < layer_size; j++)
		{
			product *= input[j];
		}

		return product;
	}
}

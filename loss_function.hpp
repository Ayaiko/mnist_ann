#pragma once

#include "cmath"
#include "vector"
#include "activation_function.hpp"
using namespace std;

namespace loss_function {

	float cross_entropy_loss(vector<float> logits)
	{
		vector<float> probabilities = activation_function::softmax(logits);

		return -log(*max_element(probabilities.begin(), probabilities.end() ) ) ;
	}

}
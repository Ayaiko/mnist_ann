#pragma once

#include "cmath"
#include "vector"
#include "activation_function.hpp"
using namespace std;

namespace loss_function {

	float cross_entropy_loss(vector<float> logits)
	{
		vector<float> probabilities = activation_function::softmax(logits);

		cout << "prob " << *max_element(probabilities.begin(), probabilities.end()) << endl;
		return -log(*max_element(probabilities.begin(), probabilities.end() ) ) ;
	}

	float cross_entropy_loss_grad(vector<float>& probabilities)
	{
		return -1.0/ (*max_element(probabilities.begin(), probabilities.end() ));
	}
}
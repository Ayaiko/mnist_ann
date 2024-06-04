#pragma once
#include <string>
#include <cmath>
#include "inspection.hpp"

using namespace std;

namespace activation_function {
	vector<float> weighted_sum(vector<vector<float>> weight, vector<float> input)
	{
		//number of neuron in the current layer
		const size_t num_neuron = weight.size();
		//number of neuron layer before
		const size_t connecting_neuron = input.size();
		vector<float> output(num_neuron, 0.0f);

		for (size_t i = 0; i < num_neuron; i++)
		{
			for (size_t j = 0; j < connecting_neuron; j++)
			{
				output[i] += weight[i][j] * input[j];
			}
		}

		return output;
	}

	vector<float> relu(vector<float> input)
	{
		const size_t size = input.size();
		vector<float> output(size);
		for (int i = 0; i < size; i++)
		{
			output[i] = max(0.0f, input[i]);
		}

		return output;
	}

	//**needed implementation
	vector<float> softmax(vector<float> input)
	{
		const size_t size = input.size();
		vector<float> output(size, 0);
		float max_input = *max_element(input.begin(), input.end());
		float sum = 0.0f;

		for (int i = 0; i < size; i++)
		{
			output[i] = exp(input[i] - max_input);
			sum += output[i];
		}

		for (int i = 0; i < size; i++)
		{
			output[i] /= sum;
		}

		return output;
	}

	vector<float> apply_activation(string activation_function, vector<vector<float>> weight, vector<float> input)
	{
		vector<float> logits;
		logits = weighted_sum(weight, input);

		if (activation_function == "relu")
		{
			return relu(logits);
		}
		else if (activation_function == "softmax")
		{
			return softmax(logits);
		}
	
		return {};
	}
}



#pragma once
#include <string>
#include <cmath>
#include "inspection.hpp"

using namespace std;

namespace activation_function {
	vector<float> relu_grad(vector<float>& logits);
	vector<float> softmax_grad(vector<float>& predicted_probabilities, float& loss_gradients);

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

	vector<float> apply_activation(string activation_function, vector<float>& logits)
	{
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

	vector<float> apply_activation_grad(string activation_function, vector<float>& logits, float& loss_gradient)
	{
		if (activation_function == "relu")
		{
			return relu_grad(logits);
		}
		else if (activation_function == "softmax")
		{
			return softmax_grad(logits, loss_gradient);
		}

	}

	vector<float> relu_grad(vector<float>& logits)
	{
		const size_t size = logits.size();
		vector<float> output(size);
		for (int i = 0; i < size; i++)
		{
			if (logits[i] > 0) output[i] = 1;
			else output[i] = 0;
		}
		return output;
	}

	vector<float> softmax_grad(vector<float>& predicted_probabilities, float& loss_gradients)
	{
		const size_t size = predicted_probabilities.size();
		vector<float> output(size);
		vector<float>::iterator max_idx = max_element(predicted_probabilities.begin(), predicted_probabilities.end() ) ;
		
		for (int i = 0; i < size; i++)
		{
			if (i == predicted_probabilities[std::distance(predicted_probabilities.begin(), max_idx)])
			{
				output[i] = loss_gradients * (*max_idx) * (1 - (*max_idx) );
			}
			else
			{
				output[i] = loss_gradients * (*max_idx) * -predicted_probabilities[i];
			}
		}


		return output;
	}
}




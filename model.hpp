#pragma once
#include "layer.hpp"
#include <iostream>
#include <vector>
#include <string>
#include "activation_function.hpp"
#include "loss_function.hpp"

using namespace std;

class Model 
{
private:
	//data members
	vector<Layer*> sequence;
	vector<float> output_label;
	vector<float> predicted_probabilities;
	vector<uint8_t> model_label;

	//helper method
	vector<float> forward_pass(vector<float>& input, string output = "None")
	{
		const size_t layer_size = sequence.size();
		vector<float> layer_output = sequence[0]->forward_propagation(input);
		for (int i = 1; i < layer_size; i++)//traverse each layer
		{
			if (output == "raw" && i == layer_size - 1) return sequence[i]->forward_propagation(layer_output, "raw");
			//compute the output of current layer from input of layer before
			layer_output = sequence[i]->forward_propagation(layer_output);
		}

		return layer_output;
	}

	void backward_pass()
	{
		float loss_grad = loss_function::cross_entropy_loss_grad(predicted_probabilities);
		size_t layer_index = sequence.size() - 1;
		vector<vector<float>> gradients(1, vector<float>(1, loss_grad) );

		for (int i = layer_index ; i > 0; i--)
		{
			gradients = sequence[i]->backward_propagation(gradients);
			sequence[i]->weight_update(gradients, 0.01f);
		}
	}

	template<typename T = uint8_t>
	vector<float> convert_to_float(vector<T>& data)
	{
		if (is_convertible<T, float>::value)
		{
			return vector<float>(data.begin(), data.end());
		}
		else
		{
			throw runtime_error("The vector type is not convertible to float.");
		}
	}

public:
	void add_layer(Layer* layer) 
	{
		sequence.push_back(layer);
	}

	void fit( vector<vector<uint8_t>>& data, vector<uint8_t>& label) 
	{
		//
		float loss_value;
		vector<float> losses;

		vector<float> model_output;
		size_t data_size = data.size();
		//store label
		model_label = label;

		while (data_size-- > 0) //traverse dataset
		{
			//convert type to float
			vector<float> input = convert_to_float(data[data_size]);
			//logit from output layer
			model_output = forward_pass(input, "raw");
			predicted_probabilities = activation_function::softmax(model_output);

			loss_value = loss_function::cross_entropy_loss(model_output);

			cout << "loss value: " << loss_value << endl;
			
			backward_pass();
		}

	}

	vector<float> predict(vector<vector<uint8_t>>& data)
	{
		vector<float> model_output;
		size_t data_size = data.size();
		while (data_size-- > 0) //traverse dataset
		{
			//convert type to float
			vector<float> input = convert_to_float(data[data_size]);
			model_output = forward_pass(input);
			model_output = activation_function::softmax(model_output);
			//store output from the output layer
			output_label[data_size] = *max_element(model_output.begin(), model_output.end());
		}

		return output_label;
	}

	vector<float> debug() const
	{
		return predicted_probabilities;
	}

	vector<float> evaluate()
	{
		return {};
	}
};
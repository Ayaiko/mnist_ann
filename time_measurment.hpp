#pragma once
#include <chrono>
#include <iostream>

namespace tmeasure
{
	void duration_measure(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end)
	{
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		std::cout << "Time duration is: " << duration.count() << std::endl;
	}
}
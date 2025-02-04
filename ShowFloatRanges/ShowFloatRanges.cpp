
#include <iostream>
#include <sstream>
#include <iomanip>
#include "../include/fltu.h"

int main(int argc, char** argv)
{
	if (argc <= 1)
		output_float_ranges();
	else if (argv[1] == std::string("--csv"))
		output_float_ranges_as_csv();
	else
		std::cerr << "ShowFloatRanges [--csv]";
	return 0;
}

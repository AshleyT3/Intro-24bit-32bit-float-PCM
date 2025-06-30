/*
 * Copyright 2024-2025 Ashley R. Thomas
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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

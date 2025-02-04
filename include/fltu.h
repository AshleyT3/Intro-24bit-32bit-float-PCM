#pragma once

#include <cstdint>
#include <cmath>
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>
#include <algorithm>

const int out_precision = 9;

const float f_to_i24bit = (float)0x7FFFFF;            // Convert float32 sample to 24-bit sample integer.
const float f_to_i32bit = (float)0x7FFFFFFF;          // Convert float32 sample to 32-bit sample integer.
const float i32bit_to_f = 1.0 / ((float)0x80000000);  // Convert 32-bit sample to float32 sample.
const float i24bit_to_f = 1.0 / ((float)0x800000);    // Convert 24-bit sample to float32 sample.


int32_t float_to_24bit(float f)
{
	// In python, precision is as if 'f * f_to_i24bit' is calculated
	// with double-precision, then passed to 'lrint' and not 'lrintf'.
	//return lrintf(f * f_to_i24bit);

	return lrint((double)f * (double)f_to_i24bit);
}


union fltu {
	float f;
	struct
	{
		unsigned int man : 23;
		unsigned int biased_exp : 8;
		unsigned int sign : 1;
	} p;
	int32_t raw;

	int get_exp() const
	{
		return std::max((signed int)p.biased_exp - 127, -126);
	}

	operator float() const
	{
		return f;
	}
	fltu(float f)
	{
		this->f = f;
	}
	fltu(unsigned int sign, unsigned int biased_exp, unsigned int man)
	{
		this->p.sign = sign;
		this->p.biased_exp = biased_exp;
		this->p.man = man;
	}
};


struct float_range
{
	fltu low_end;
	fltu low_end_plus_1;
	//...
	fltu high_end_minus_1;
	fltu high_end;

	float_range(int biased_exp)
		:
		low_end(0, biased_exp, 0),
		low_end_plus_1(0, biased_exp, 1),
		high_end_minus_1(0, biased_exp, -2),
		high_end(0, biased_exp, -1)
	{
	}
};


std::vector<float_range> get_float_ranges()
{
	std::vector<float_range> v;
	for (int exp = -127; exp <= 0; exp++) {
		int biased_exp = 127 + (char)exp; // unbiased exp == biased_exp - 127 
		v.push_back(float_range(biased_exp));
	}
	return v;
}


const std::string get_fltu_log_str(const char* msg, int msg_padding, const fltu& u)
{
	std::ostringstream os;

	if (msg && *msg)
	{
		std::ostringstream os_msg;
		os_msg << msg << " ";
		os << std::left << std::setw(msg_padding) << std::setfill('.');
		os << os_msg.str();
		os << " " << std::right;
	}

	os << std::scientific << std::setprecision(out_precision) << u.f;
	os << " ";
	os << "(" << std::fixed << std::setprecision(out_precision) << u.f << ")";
	os << " ";
	os << "("
		<< "sign=" << u.p.sign
		<< " bexp=" << u.p.biased_exp << " exp=" << u.get_exp()
		<< " man=0x" << std::hex << std::setw(6) << std::setfill('0') << u.p.man
		<< " raw=0x" << std::hex << std::setw(8) << std::setfill('0') << u.raw
		<< ")"
		<< " "
		<< "("
		<< "24bit: " << "0x" << std::hex << std::setw(6) << std::setfill('0') << (float_to_24bit(u.f) & 0xFFFFFF)
		<< " dec=" << std::dec << float_to_24bit(u.f)
		<< ")";
	return os.str();
}


const std::string get_fltu_log_str(const fltu& u)
{
	return get_fltu_log_str(nullptr, 0, u);
}



void output_float_ranges()
{
	for (auto& fr : get_float_ranges())
	{
		std::cout << std::scientific << std::setprecision(out_precision) << fr.low_end.f << " to ";
		std::cout << std::scientific << std::setprecision(out_precision) << fr.high_end.f;
		std::cout << std::endl;
		int padding = 14;
		std::cout << "     " << get_fltu_log_str("low_end", padding, fr.low_end) << std::endl;
		std::cout << "     " << get_fltu_log_str("low_end+1", padding, fr.low_end_plus_1) << std::endl;
		std::cout << "     ... " << std::endl;
		std::cout << "     " << get_fltu_log_str("high_end-1", padding, fr.high_end_minus_1) << std::endl;
		std::cout << "     " << get_fltu_log_str("high_end", padding, fr.high_end) << std::endl;
		std::cout << std::endl;
	}
}


std::string get_fltu_csv_heaader_part(const std::string& name_pfx)
{
	std::ostringstream os;
	os << name_pfx
		<< ","
		<< name_pfx << "_raw"
		<< ","
		<< name_pfx << "_sign"
		<< ","
		<< name_pfx << "_biased_exp"
		<< ","
		<< name_pfx << "_mantissa"
		<< ","
		<< name_pfx << "_24bit"
		<< ","
		<< name_pfx << "_24bit_hex";
	return os.str();
}


std::string get_fltu_csv_part(const fltu& fltu_val)
{
	std::ostringstream os;
	os << std::scientific
		<< std::setprecision(out_precision)
		<< fltu_val.f
		<< ","
		<< "0x" << std::hex << std::setw(8) << std::setfill('0') << fltu_val.raw << std::dec
		<< ","
		<< fltu_val.p.sign
		<< ","
		<< fltu_val.p.biased_exp
		<< ","
		<< fltu_val.p.man
		<< ","
		<< std::dec << float_to_24bit(fltu_val.f)
		<< ","
		<< "0x" << std::hex << std::setw(6) << std::setfill('0') << (float_to_24bit(fltu_val.f) & 0xFFFFFF) << std::dec;
	return os.str();
}


void output_float_ranges_as_csv()
{
	std::ostringstream os;
	os << get_fltu_csv_heaader_part("low_end")
		<< ","
		<< get_fltu_csv_heaader_part("low_end_plus_1")
		<< ","
		<< get_fltu_csv_heaader_part("high_end_minus_1")
		<< ","
		<< get_fltu_csv_heaader_part("high_end");
	std::cout << os.str() << std::endl;

	for (auto& fr : get_float_ranges())
	{
		os.str("");
		os << get_fltu_csv_part(fr.low_end)
			<< ","
			<< get_fltu_csv_part(fr.low_end_plus_1)
			<< ","
			<< get_fltu_csv_part(fr.high_end_minus_1)
			<< ","
			<< get_fltu_csv_part(fr.high_end);
		std::cout << os.str() << std::endl;
	}
}

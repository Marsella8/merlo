#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>
#include "matrix.h"

void not_implemented_impl(const char* file, int line);
void trace_impl(const char* label, const char* file, int line, const char* fmt, ...);
void warning_impl(const char* file, int line, const char* message);
void assume_no_nans_impl(Matrix m, const char* file, int line);

#define not_implemented() not_implemented_impl(__FILE__, __LINE__)
#define trace_head(...) trace_impl("HEAD", __FILE__, __LINE__, __VA_ARGS__)
#define trace_tail(...) trace_impl("PIPE", __FILE__, __LINE__, __VA_ARGS__)
#define warning(message) warning_impl(__FILE__, __LINE__, (message))
#define assume_no_nans(m) assume_no_nans_impl((m), __FILE__, __LINE__)

size_t ceil_div(size_t x, size_t y);
char* substr(const char* str, size_t start, size_t end); // make sure to free!
#endif // UTILS_H

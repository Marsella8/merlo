#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

void panic_impl(const char* file, int line, const char* message);
void not_implemented_impl(const char* file, int line);
void warning_impl(const char* file, int line, const char* message);
void assume_no_nans_impl(Matrix m, const char* file, int line);

#define panic(message) panic_impl(__FILE__, __LINE__, (message))
#define not_implemented() not_implemented_impl(__FILE__, __LINE__)
#define warning(message) warning_impl(__FILE__, __LINE__, (message))
#define assume_no_nans(m) assume_no_nans_impl((m), __FILE__, __LINE__)

char* substr(const char* str, size_t start, size_t end); // make sure to free!
#endif // UTILS_H

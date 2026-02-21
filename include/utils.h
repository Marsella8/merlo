#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>

void panic_impl(const char* file, int line, const char* message);
void not_implemented_impl(const char* file, int line);
void warning_impl(const char* file, int line, const char* message);

#define panic(message) panic_impl(__FILE__, __LINE__, (message))
#define not_implemented() not_implemented_impl(__FILE__, __LINE__)
#define warning(message) warning_impl(__FILE__, __LINE__, (message))

char* substr(const char* str, size_t start, size_t end); // make sure to free!
#endif // UTILS_H

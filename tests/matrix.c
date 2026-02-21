#include <assert.h>
#include <stdio.h>
#include <math.h>
#include "matrix.h"

float m[2][3] = {
    {1.0f, 2.0f, 3.0f},
    {4.0f, 5.0f, 6.0f}
};

void test_transpose_shares_buffer() {
    printf("  test_transpose_shares_buffer\n");
    Matrix matrix = mat_from_array(2, 3, m);
    Buffer* buf = matrix.buffer;
    Matrix transposed = transpose(matrix);
    assert(buf == transposed.buffer);
    free_mat(matrix);
}

void test_at() {
    printf("  test_at\n");
    Matrix matrix = mat_from_array(2, 3, m);
    
    assert(*at(matrix, 0, 0) == 1.0f);
    assert(*at(matrix, 0, 1) == 2.0f);
    assert(*at(matrix, 0, 2) == 3.0f);
    assert(*at(matrix, 1, 0) == 4.0f);
    assert(*at(matrix, 1, 1) == 5.0f);
    assert(*at(matrix, 1, 2) == 6.0f);
    
    free_mat(matrix);
}

void test_transpose() {
    printf("  test_transpose\n");
    float m_data[2][3] = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f}
    };
    float correct_data[3][2] = {
        {1.0f, 4.0f},
        {2.0f, 5.0f},
        {3.0f, 6.0f}
    };

    Matrix a = mat_from_array(2, 3, m_data);
    Matrix result = transpose(a);
    Matrix correct = mat_from_array(3, 2, correct_data);
    
    assert(eq(result, correct));

    free_mat(a);
    free_mat(correct);
}

void test_add() {
    printf("  test_add\n");
    float a_data[2][2] = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    float b_data[2][2] = {{5.0f, 6.0f}, {7.0f, 8.0f}};
    float correct_data[2][2] = {{6.0f, 8.0f}, {10.0f, 12.0f}};
    
    Matrix a = mat_from_array(2, 2, a_data);
    Matrix b = mat_from_array(2, 2, b_data);
    Matrix result = add(a, b);
    Matrix correct = mat_from_array(2, 2, correct_data);
    
    assert(eq(result, correct));
    
    free_mat(a);
    free_mat(b);
    free_mat(result);
    free_mat(correct);
}

void test_matmul() {
    printf("  test_matmul\n");
    float a_data[2][3] = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    float b_data[3][2] = {{7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 12.0f}};
    float correct_data[2][2] = {
        {1.0f*7.0f + 2.0f*9.0f + 3.0f*11.0f, 1.0f*8.0f + 2.0f*10.0f + 3.0f*12.0f},
        {4.0f*7.0f + 5.0f*9.0f + 6.0f*11.0f, 4.0f*8.0f + 5.0f*10.0f + 6.0f*12.0f}
    };
    
    Matrix a = mat_from_array(2, 3, a_data);
    Matrix b = mat_from_array(3, 2, b_data);
    Matrix result = matmul(a, b);
    Matrix correct = mat_from_array(2, 2, correct_data);
    
    assert(eq(result, correct));
    
    free_mat(a);
    free_mat(b);
    free_mat(result);
    free_mat(correct);
}

void test_slice() {
    printf("  test_slice\n");
    float m_data[3][3] = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f},
        {7.0f, 8.0f, 9.0f}
    };
    Matrix a = mat_from_array(3, 3, m_data);
    
    {
        float correct_data[2][2] = {{5.0f, 6.0f}, {8.0f, 9.0f}};
        Matrix correct = mat_from_array(2, 2, correct_data);
        Matrix s = slice(a, 1, 3, 1, 3);
        assert(eq(s, correct));
        free_mat(correct);
    }

    {
        float correct_data[1][3] = {{1.0f, 2.0f, 3.0f}};
        Matrix correct = mat_from_array(1, 3, correct_data);
        Matrix s = slice(a, 0, 1, 0, 3);
        assert(eq(s, correct));
        free_mat(correct);
    }

    {
        float correct_data[3][1] = {{3.0f}, {6.0f}, {9.0f}};
        Matrix correct = mat_from_array(3, 1, correct_data);
        Matrix s = slice(a, 0, 3, 2, 3);
        assert(eq(s, correct));
        free_mat(correct);
    }

    free_mat(a);
}

void test_masked_matmul() {
    printf("  test_masked_matmul\n");
    float a_data[3][3] = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f},
        {7.0f, 8.0f, 9.0f}
    };
    float b_data[3][3] = {
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f}
    };
    Matrix a = mat_from_array(3, 3, a_data);
    Matrix b = mat_from_array(3, 3, b_data);
    Matrix result = masked_matmul(a, b);
    assert(*at(result, 0, 0) == 1.0f);
    assert(isinf(*at(result, 0, 1)) && *at(result, 0, 1) < 0.0f);
    assert(isinf(*at(result, 0, 2)) && *at(result, 0, 2) < 0.0f);
    assert(*at(result, 1, 0) == 4.0f);
    assert(*at(result, 1, 1) == 5.0f);
    assert(isinf(*at(result, 1, 2)) && *at(result, 1, 2) < 0.0f);
    assert(*at(result, 2, 0) == 7.0f);
    assert(*at(result, 2, 1) == 8.0f);
    assert(*at(result, 2, 2) == 9.0f);

    free_mat(a);
    free_mat(b);
    free_mat(result);
}


int main() {
    printf("matrix tests:\n");
    test_at();
    test_transpose();
    test_transpose_shares_buffer();
    test_add();
    test_matmul();
    test_masked_matmul();
    test_slice();
    return 0;
}

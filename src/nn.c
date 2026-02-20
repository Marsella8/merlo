#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "nn.h"
#include "model.h"
#include "utils.h"

Matrix silu(Matrix x) {
    Buffer* b = buf(x.rows * x.cols * sizeof(float));
    Matrix m = mat(b, x.rows, x.cols);
    
    for (size_t i = 0; i < x.rows; i++) {
        for (size_t j = 0; j < x.cols; j++) {
            float val = *at(x, i, j);
            float sigmoid = 1.0f / (1.0f + expf(-val));
            *at(m, i, j) = val * sigmoid;
        }
    }
    
    return m;
}


Matrix embed(Matrix embeddings, Matrix tokens) {
#ifdef SAFETY
    assert(tokens.rows == 1);
#endif
    Buffer* b = buf(tokens.cols * embeddings.cols * sizeof(float));
    Matrix m = mat(b, tokens.cols, embeddings.cols);
    for (size_t i = 0; i < tokens.cols; i++) {
        size_t id = (size_t)*at(tokens, 0, i);
        memcpy(at(m, i, 0), at(embeddings, id, 0), embeddings.cols * sizeof(float));
    }
    return m;
}

Matrix rope(Matrix x, int pos) {
    return x;
}

Matrix rms_norm(Matrix x, Matrix weight) {
    Buffer* b = buf(x.rows * x.cols * sizeof(float));
    Matrix m = mat(b, x.rows, x.cols);
    for (size_t i=0; i<x.rows; i++) {
        float rms = 0;
        for (size_t j=0; j<x.cols; j++) {
            rms+=*at(x,i,j) * *at(x,i,j);
        }
        rms = rms/x.cols;
        rms += RMS_NORM_EPS;
        float i_rms = 1.0f / sqrtf(rms); // TODO: sub with the cool quake III algo
        for (size_t j=0; j<x.cols; j++) {
            *at(m, i, j) = *at(x, i, j) * *at(weight, 0, j) * i_rms;
        }
    }
    return m;
}

Matrix engine_gqa(Matrix x, Matrix q, Matrix k, Matrix v, Matrix o, LayerCache* cache, int pos) {
#ifdef SAFETY
    assert(x.rows==1 && x.cols==HIDDEN_SIZE);
    assert(q.rows > 0);
#endif
    panic("Not implemented");
    return x;
}

Matrix prefill_gqa(Matrix x, Matrix q, Matrix k, Matrix v, Matrix o, int pos, LayerCache* cache) {
    panic("Not implemented");
    return x;
}

Matrix elementwise(Matrix a, Matrix b) {
#ifdef SAFETY
    if (a.rows != b.rows || a.cols != b.cols) {
        fprintf(stderr, "Error: Matrix dimensions mismatch in elementwise()\n");
        exit(1);
    }
#endif
    
    Buffer* buf_out = buf(a.rows * a.cols * sizeof(float));
    Matrix m = mat(buf_out, a.rows, a.cols);
    
    for (size_t i = 0; i < a.rows; i++) {
        for (size_t j = 0; j < a.cols; j++) {
            *at(m, i, j) = *at(a, i, j) * *at(b, i, j);
        }
    }
    
    return m;
}

Matrix ffn(Matrix x, Matrix gate, Matrix up, Matrix down) {
    Matrix m = matmul(x, gate);
    Matrix xgate = silu(m);
    free_buf(m.buffer);

    Matrix xup = matmul(x, up);
    Matrix ew = elementwise(xgate, xup);
    free_buf(xgate.buffer);
    free_buf(xup.buffer);
    
    Matrix out = matmul(ew, down);
    free_buf(ew.buffer);
    
    return out;
}

Matrix softmax(Matrix x) {
    Buffer* b = buf(x.rows * x.cols * sizeof(float));
    Matrix m = mat(b, x.rows, x.cols);
    
    for (size_t i = 0; i < x.rows; i++) {
        float max_val = *at(x, i, 0);
        for (size_t j = 1; j < x.cols; j++) {
            float val = *at(x, i, j);
            if (val > max_val) max_val = val;
        }
        
        float sum = 0.0f;
        for (size_t j = 0; j < x.cols; j++) {
            float exp_val = expf(*at(x, i, j) - max_val);
            *at(m, i, j) = exp_val;
            sum += exp_val;
        }
        
        for (size_t j = 0; j < x.cols; j++) {
            *at(m, i, j) /= sum;
        }
    }
    
    return m;
}
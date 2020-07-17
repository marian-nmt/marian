/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_VECTOR_TRANSFORM_H
#define FAISS_VECTOR_TRANSFORM_H

/** Defines a few objects that apply transformations to a set of
 * vectors Often these are pre-processing steps.
 */

#include <vector>
#include <stdint.h>

#include <faiss/Index.h>
#ifdef __APPLE__
#include <x86intrin.h>
#endif


namespace faiss {


/** Any transformation applied on a set of vectors */
struct VectorTransform {

    typedef Index::idx_t idx_t;

    int d_in;      ///! input dimension
    int d_out;     ///! output dimension

    explicit VectorTransform (int d_in = 0, int d_out = 0):
    d_in(d_in), d_out(d_out), is_trained(true)
    {}


    /// set if the VectorTransform does not require training, or if
    /// training is done already
    bool is_trained;


    /** Perform training on a representative set of vectors. Does
     * nothing by default.
     *
     * @param n      nb of training vectors
     * @param x      training vecors, size n * d
     */
    virtual void train (idx_t n, const float *x);

    /** apply the random roation, return new allocated matrix
     * @param     x size n * d_in
     * @return    size n * d_out
     */
    float *apply (idx_t n, const float * x) const;

    /// same as apply, but result is pre-allocated
    virtual void apply_noalloc (idx_t n, const float * x,
                                float *xt) const = 0;

    /// reverse transformation. May not be implemented or may return
    /// approximate result
    virtual void reverse_transform (idx_t n, const float * xt,
                                    float *x) const;

    virtual ~VectorTransform () {}

};



/** Generic linear transformation, with bias term applied on output
 * y = A * x + b
 */
struct LinearTransform: VectorTransform {

    bool have_bias; ///! whether to use the bias term

    /// check if matrix A is orthonormal (enables reverse_transform)
    bool is_orthonormal;

    /// Transformation matrix, size d_out * d_in
    std::vector<float> A;

     /// bias vector, size d_out
    std::vector<float> b;

    /// both d_in > d_out and d_out < d_in are supported
    explicit LinearTransform (int d_in = 0, int d_out = 0,
                              bool have_bias = false);

    /// same as apply, but result is pre-allocated
    void apply_noalloc(idx_t n, const float* x, float* xt) const override;

    /// compute x = A^T * (x - b)
    /// is reverse transform if A has orthonormal lines
    void transform_transpose (idx_t n, const float * y,
                              float *x) const;

    /// works only if is_orthonormal
    void reverse_transform (idx_t n, const float * xt,
                            float *x) const override;

    /// compute A^T * A to set the is_orthonormal flag
    void set_is_orthonormal ();

    bool verbose;
    void print_if_verbose (const char*name, const std::vector<double> &mat,
                           int n, int d) const;

    ~LinearTransform() override {}
};



/// Randomly rotate a set of vectors
struct RandomRotationMatrix: LinearTransform {

     /// both d_in > d_out and d_out < d_in are supported
     RandomRotationMatrix (int d_in, int d_out):
         LinearTransform(d_in, d_out, false) {}

     /// must be called before the transform is used
     void init(int seed);

     // intializes with an arbitrary seed
     void train(idx_t n, const float* x) override;

     RandomRotationMatrix () {}
};


/** Applies a principal component analysis on a set of vectors,
 *  with optionally whitening and random rotation. */
struct PCAMatrix: LinearTransform {

    /** after transformation the components are multiplied by
     * eigenvalues^eigen_power
     *
     * =0: no whitening
     * =-0.5: full whitening
     */
    float eigen_power;

    /// random rotation after PCA
    bool random_rotation;

    /// ratio between # training vectors and dimension
    size_t max_points_per_d;

    /// try to distribute output eigenvectors in this many bins
    int balanced_bins;

    /// Mean, size d_in
    std::vector<float> mean;

    /// eigenvalues of covariance matrix (= squared singular values)
    std::vector<float> eigenvalues;

    /// PCA matrix, size d_in * d_in
    std::vector<float> PCAMat;

    // the final matrix is computed after random rotation and/or whitening
    explicit PCAMatrix (int d_in = 0, int d_out = 0,
                        float eigen_power = 0, bool random_rotation = false);

    /// train on n vectors. If n < d_in then the eigenvector matrix
    /// will be completed with 0s
    void train(idx_t n, const float* x) override;

    /// copy pre-trained PCA matrix
    void copy_from (const PCAMatrix & other);

    /// called after mean, PCAMat and eigenvalues are computed
    void prepare_Ab();

};


} // namespace faiss


#endif

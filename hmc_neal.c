/* Self-contained file for Hamiltonian Monte Carlo (HMC) targeting the 100
 * dimensional Gaussian distribution appearing as an example in
 * 'MCMC using Hamiltonian Dynamics', Neal, 2011. Variables are independent.
 * The standard deviation of variable i is i/100 for i = 1,2,...,100.
 * Note: 40,000 samples took approximately 60 seconds on my laptop.
 *
 * I wrote this code to improve my understanding of HMC and the GSL Scientific
 * Library. If you want to use HMC for your own application, I recommend
 * using STAN.
 *
 * Use:
 * Requires the GSL Scientific Library.
 * Compile with (something like):
 * gcc -Wall -lgsl -lgslcblas -lm hmc_neal.c -o hmc
 *
 * Then run and save samples to a file:
 * ./hmc > samples.txt
 *
 * Then analyse with R or Python.
 *
 * N         : Number of samples
 * STEP_SIZE : Leapfrog step-size
 * N_STEPS   : Number of leapfrog steps
 */

#include <stdio.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>
#define N 40000
#define STEP_SIZE 0.013
#define N_UPDATES 150
#define DIMENSION 100

gsl_matrix * precision;
gsl_vector * work;

void print_vec_8dp(const gsl_vector * x, int d);

double pe_mvg(const gsl_vector * q);

void pe_mvg_grad(const gsl_vector * q, gsl_vector * pe_grad_q);

double ke_std(const gsl_vector * p);

void leapfrog_update(gsl_vector * q, gsl_vector * p, gsl_vector * pe_grad_q,
                      void (*pe_grad)(const gsl_vector * q,
                                      gsl_vector * pe_grad_q),
                      double step_size, int n_updates);

void hmc_kern(const gsl_rng * r, const gsl_vector * q, const gsl_vector * p,
              gsl_vector * q_new, gsl_vector * p_new, gsl_vector * pe_grad_q,
              double (*pe)(const gsl_vector * q),
              void (*pe_grad)(const gsl_vector * q, gsl_vector * pe_grad_q),
              double step_size, int n_updates);

int main(){
    int d = DIMENSION;
    int n = N;
    gsl_rng * r = gsl_rng_alloc(gsl_rng_mt19937);
    
    // Additional workspace required (global variable)
    work = gsl_vector_alloc(d);
    
    // Precision matrix for the target distribution
    precision = gsl_matrix_alloc(d, d);
    gsl_matrix_set_all(precision, 0.0);
    for (int i = 0; i < d; i++){
        gsl_matrix_set(precision, i, i, pow((i+1)/100.0, 2));
    }
    gsl_linalg_cholesky_decomp1(precision);
    gsl_linalg_cholesky_invert(precision);
    
    // Mean and Cholesky decomposition of the precision matrix of the Gaussian
    // distribution corresponding to the kinetic energy.
    gsl_vector * p_mu = gsl_vector_calloc(d);
    gsl_matrix * p_L  = gsl_matrix_alloc(d, d);
    gsl_matrix_set_identity(p_L);
    
    // Initialise position and momentum from standard Gaussian distribution.
    gsl_vector * q = gsl_vector_alloc(d);
    gsl_vector * p = gsl_vector_alloc(d);
    gsl_ran_multivariate_gaussian(r, p_mu, p_L, q);
    gsl_ran_multivariate_gaussian(r, p_mu, p_L, p);
    
    // Gradient of the potential energy at q
    gsl_vector * pe_grad_q = gsl_vector_alloc(d);
    
    // Leapfrog step-size and number of steps
    double step_size = STEP_SIZE;
    int n_updates = N_UPDATES;
    
    // Vectors for proposed q and p
    gsl_vector * q_new = gsl_vector_alloc(d);
    gsl_vector * p_new = gsl_vector_alloc(d);
    
    // Print the initial position
    print_vec_8dp(q, d);
    
    // HMC sampling
    for (int i = 0; i < n; i++){
        
        // Simulate the momentum
        gsl_ran_multivariate_gaussian(r, p_mu, p_L, p);
        
        // Apply HMC kernel
        hmc_kern(r, q, p, q_new, p_new, pe_grad_q, pe_mvg, pe_mvg_grad,
                 step_size, n_updates);
        
        // Update the current state
        gsl_vector_memcpy(q, q_new);
        gsl_vector_memcpy(p, p_new);
        
        // Print
        print_vec_8dp(q, d);
    }
    
    gsl_matrix_free(precision);
    gsl_vector_free(work);
    gsl_rng_free(r);
    gsl_vector_free(p_mu);
    gsl_matrix_free(p_L);
    gsl_vector_free(q);
    gsl_vector_free(p);
    gsl_vector_free(pe_grad_q);
    gsl_vector_free(q_new);
    gsl_vector_free(p_new);
    
    return 0;
}

// Print a vector x of length d to screen (8 decimal places)
void print_vec_8dp(const gsl_vector * x, int d){
	for (int i = 0; i < d; i++){
		printf("%.8f ", gsl_vector_get(x, i));
	}
	printf("\n");
}

/* Potential energy of a multivariate Gaussian distribution
 *
 * Global variable 'precision' is the precision matrix of the distribution.
 * Global variable 'work' used for computation.
 *
 * q : position
 *
 * Returns the potential energy at q
 */
double pe_mvg(const gsl_vector * q){
    
    // precision * q
    gsl_blas_dgemv(CblasNoTrans, 1.0, precision, q, 0.0, work);
    
    // q^T * precision * q
    double result;
    gsl_blas_ddot(q, work, &result);
    
    return 0.5 * result;
}

/* Gradient of the energy of the multivariate Gaussian target distribution
 *
 * Global variable 'precision' is the precision matrix of the distribution.
 *
 * q         : Position
 * pe_grad_q : Gradient of the energy
 *
 * Updates pe_grad_q
 */
void pe_mvg_grad(const gsl_vector * q, gsl_vector * pe_grad_q){
    
    gsl_blas_dgemv(CblasNoTrans, 1.0, precision, q, 0.0, pe_grad_q);
}

/* Standard Kinetic Energy
 *
 * Corresponds to a standard Gaussian distribution,
 * K(p) = 0.5 * <p,p> for <,> the dot product.
 *
 * p 			: momentum
 * ke_precision : Precision matrix for the kinetic energy
 * work 		: Additional workspace (same length as q)
 *
 * Returns the value of the kinetic energy at p
 */
double ke_std(const gsl_vector * p){
	
	double result;
	gsl_blas_ddot(p, p, &result);
	
	return 0.5 * result;
}

/* Leapfrog updates of position and momentum
 *
 * q               : position
 * p               : momentum
 * pe_grad_q       : Gradient of the potential energy at q
 * pe_grad         : Function to compute the gradient of the potential energy
 * step_size       : Leapfrog step size
 * n_updates       : Number of leapfrog updates
 */
void leapfrog_update(gsl_vector * q, gsl_vector * p, gsl_vector * pe_grad_q,
                      void (*pe_grad)(const gsl_vector * q,
                                      gsl_vector * pe_grad_q),
                      double step_size, int n_updates){
    
    // Half step for momentum at beginning
    pe_grad(q, pe_grad_q);
    gsl_blas_dscal(0.5 * step_size, pe_grad_q);
    gsl_vector_sub(p, pe_grad_q);
    
    // Alternate full steps for position and momentum
    for (int i = 0; i < n_updates; i++){
        // Full step for the position
        gsl_blas_dscal(step_size, p);
        gsl_vector_add(q, p);
        gsl_blas_dscal(1.0/step_size, p); // Undo scaling of p
        
        // Make a full step for the momentum, except at end of trajectory
        if (i < (n_updates-1)){
            pe_grad(q, pe_grad_q);
            gsl_blas_dscal(step_size, pe_grad_q);
            gsl_vector_sub(p, pe_grad_q);
        }
    }
    
    // Make a half step for the momentum at the end
    pe_grad(q, pe_grad_q);
    gsl_blas_dscal(0.5 * step_size, pe_grad_q);
    gsl_vector_sub(p, pe_grad_q);
}

/* Hamiltonian Monte Carlo kernel
 *
 * Efficient Leapfrog updates. Kinetic energy corresponding to a standard
 * Gaussian.
 *
 * r : Random number generator
 * q : Position
 * p : Momentum
 * q_new : New position
 * p_new : New momentum
 * pe_grad_q : Gradient of the potential energy at q
 * work : Additional workspace required
 * pe : Function to compute the potential energy
 * pe_grad: Function to compute the gradient of the potential energy
 * ke : Function to compute the kinetic energy (dimension-dependent)
 * step_size : Leapfrog step-size
 * n_updates : Number of leapfrog updates
 */
void hmc_kern(const gsl_rng * r, const gsl_vector * q, const gsl_vector * p,
              gsl_vector * q_new, gsl_vector * p_new, gsl_vector * pe_grad_q,
              double (*pe)(const gsl_vector * q),
              void (*pe_grad)(const gsl_vector * q, gsl_vector * pe_grad_q),
              double step_size, int n_updates){
    
    // Make proposal
    gsl_vector_memcpy(q_new, q);
    gsl_vector_memcpy(p_new, p);
    leapfrog_update(q_new, p_new, pe_grad_q, pe_grad, step_size, n_updates);
    
    // Acceptence Probability
    double log_aprob = pe(q) + ke_std(p) - pe(q_new) - ke_std(p_new);
    
    // Accept or reject
    double u = gsl_ran_flat(r, 0.0, 1.0);
    if (log(u) > log_aprob){
        // Reject
        gsl_vector_memcpy(q_new, q);
        gsl_vector_memcpy(p_new, p);
    }
}


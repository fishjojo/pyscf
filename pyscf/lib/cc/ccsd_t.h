#ifndef HAVE_DEFINED_CCSD_T_H
#define HAVE_DEFINED_CCSD_T_H

typedef struct {
        void *cache[6];
        short a;
        short b;
        short c;
        short _padding;
} CacheJob;

size_t _ccsd_t_gen_jobs(CacheJob *jobs, int nocc, int nvir,
                        int a0, int a1, int b0, int b1,
                        void *cache_row_a, void *cache_col_a,
                        void *cache_row_b, void *cache_col_b, size_t stride);

void _make_permute_indices(int *idx, int n);

double _ccsd_t_get_energy(double *w, double *v, double *mo_energy, int nocc,
                          int a, int b, int c, double fac);

double _ccsd_t_zget_energy(double complex *w, double complex *v,
                           double *mo_energy, int nocc,
                           int a, int b, int c, double fac);

#endif

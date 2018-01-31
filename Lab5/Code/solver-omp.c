#include "heat.h"



/*
 * Function to copy one matrix into another
 */

void copy_mat (double *u, double *v, unsigned sizex, unsigned sizey)
{
    #pragma omp for collapse(2)
    for (int i=1; i<=sizex-2; i++)
        for (int j=1; j<=sizey-2; j++) 
            v[ i*sizey+j ] = u[ i*sizey+j ];
}

/*
 * Blocked Jacobi solver: one iteration step
 */
double relax_jacobi (double *u, double *utmp, unsigned sizex, unsigned sizey)
{
    double diff, sum=0.0;
  
    int howmany= omp_get_max_threads();
    #pragma omp parallel for private(diff) reduction (+:sum)
    for (int blockid = 0; blockid < howmany; ++blockid) {
      int i_start = lowerb(blockid, howmany, sizex);
      int i_end = upperb(blockid, howmany, sizex);
    //printf("id[%d] \t", blockid);
      for (int i=max(1, i_start); i<= min(sizex-2, i_end); i++) {
        for (int j=1; j<= sizey-2; j++) {
         utmp[i*sizey+j]= 0.25 * ( u[ i*sizey     + (j-1) ]+  // left
                                   u[ i*sizey     + (j+1) ]+  // right
                       u[ (i-1)*sizey + j     ]+  // top
                       u[ (i+1)*sizey + j     ]); // bottom
         diff = utmp[i*sizey+j] - u[i*sizey + j];
         sum += diff * diff; 
     }
      }
    }
    return sum;
}



/*
 * Blocked Gauss-Seidel solver: one iteration step
*/
double relax_gauss (double *u, unsigned sizex, unsigned sizey){
    double unew, diff, sum=0.0;
    int howmany = omp_get_max_threads();
    int procesados[howmany];

    #pragma omp parallel for
    for (int i = 0; i < howmany; ++i) {
        procesados[i] = 0;
    }
    int nb = 8;

    #pragma omp parallel for schedule(static) private(diff, unew) reduction(+ : sum)
    for (int i = 0; i < howmany; ++i) {
        int iilowerb = lowerb(i, howmany, sizex);
        int iiupperb = upperb(i, howmany, sizex);
        for (int j = 0; j < nb; j++){
            int jjlowerb = lowerb(j,nb,sizey);
            int jjupperb = upperb(j,nb,sizey);
            if (i > 0){
                while (procesados[i-1] <= j){
                #pragma omp flush
                }
            }

            for (int ii = max(1, iilowerb); ii <= min(sizex-2, iiupperb); ii++) {
                for (int jj= max(1, jjlowerb); jj <= min(sizey-2, jjupperb); jj++){
                    unew = 0.25* (u[ii * sizey + (jj-1)] +  // left
                                  u[ii * sizey + (jj+1)] +  // right
                                  u[(ii-1) * sizey + jj] +  // top
                                  u[(ii+1) * sizey + jj]); // bottom
                    diff = unew - u[ii * sizey + jj];
                    sum += diff*diff;
                    u[ii*sizey+jj] = unew;
                }
            }
            ++procesados[i];
            #pragma omp flush
        }
    }
    return sum;
}







/*
 * Blocked Gauss-Seidel solver OPTIONAL: one iteration step
* /

double relax_gauss(double *u, unsigned sizex, unsigned sizey) {
    double unew, diff, sum=0.0;
    int howmany = omp_get_max_threads();

    //Block number
    int nb = 8;

    // Aux matrix for dependencies, data doesn't matter (use char for less space used)
    char auxDeps[howmany][howmany];

    #pragma omp parallel private(diff, unew)
    #pragma omp single
    for (int i = 0; i < howmany; ++i) {
        int iilowerb = lowerb(i, howmany, sizex); //Lower bound for ii
        int iiupperb = upperb(i, howmany, sizex); //Lower bound for jj
        for (int j = 0; j < nb; j++){
            int jjlowerb = lowerb(j, nb, sizey);
            int jjupperb = upperb(j, nb, sizey);

        
        #pragma omp task depend(in: auxDeps[i-1][j])  depend(in: auxDeps[i][j-1])  depend(out: auxDeps[i][j])
        {
            double sum2 = 0.0;
            for (int ii = max(1, iilowerb); ii <= min(sizex-2, iiupperb); ii++) {
                    for (int jj = max(1, jjlowerb); jj <= min(sizey-2, jjupperb); jj++){
                        unew = 0.25 * (u[ii * sizey + (jj-1)] +  // left
                                      u[ii * sizey + (jj+1)] +  // right
                                      u[(ii-1) * sizey + jj] +  // top
                                      u[(ii+1) * sizey + jj]); // bottom
                        diff = unew - u[ii * sizey + jj];
                        sum2 += diff*diff;
                        u[ii*sizey+jj] = unew;
                    }
                }
                #pragma omp atomic
                sum += sum2;
            }
        }
    }
    return sum;
}
*/
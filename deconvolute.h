#ifndef _DECONVOLUTE_H_
#define _DECONVOLUTE_H_

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <fftw3.h>
#include <CL/opencl.h>
#include "tiff_goodness.h"
#include "opencl_utils.h"

int deconvolute_image(char *input_image_filename, char
		*psf_image_filename, char *output_image_filename, int
		n_iterations);

#endif /* _DECONVOLUTE_H_ */

#include <stdlib.h>
#include <stdio.h>
#include "deconvolute.h"

#define OUTPUT_FILENAME "deconvoluted_image.tif"
int main(int argc, char *argv[])
{
	if (argc != 4) {
		fprintf(stderr, "Usage: deconvolute [input 16-bit TIFF image] [psf 8-bit TIFF image] [number of iterations]\n");
		fflush(stderr);
		return EXIT_FAILURE;
	}

	if (deconvolute_image(argv[1], argv[2], OUTPUT_FILENAME,
			atoi(argv[3]), 8) != 0) {
		return EXIT_FAILURE;
	}

	return 0;
}

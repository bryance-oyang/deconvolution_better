#include "emalloc.h"

/* error checking malloc */
void *emalloc(size_t bytes)
{
	void *p;

	if ((p = malloc(bytes)) == NULL) {
		fprintf(stderr, "%s\n", strerror(ENOMEM));
		fflush(stderr);
		exit(EXIT_FAILURE);
	}

	return p;
}

EXEC=deconvolute
srcdir=

SHELL=/bin/sh
CC=cc -pipe -mtune=native -march=native
OFLAGS=-O2
CFLAGS+=-std=c11 -Wall -pedantic-errors
LDFLAGS=-lc -lOpenCL -ltiff -lfftw3
CDEBUG=-g -p
DFLAGS=$(CFLAGS) -M

ifdef srcdir
VPATH=$(srcdir)
SRCS=$(wildcard $(srcdir)/*.c)
HDRS=$(wildcard $(srcdir)/*.h)
CFLAGS+=-I. -I$(srcdir)
else
SRCS=$(wildcard *.c)
HDRS=$(wildcard *.h)
endif
OBJS=$(SRCS:.c=.o)
DEPS=$(SRCS:.c=.d)

ifeq ($(MAKECMDGOALS), debug)
CFLAGS+=$(CDEBUG)
else
CFLAGS+=$(OFLAGS)
LDFLAGS+=-Wl,$(OFLAGS)
endif

ifneq ($(MAKECMDGOALS), clean)
-include $(DEPS)
endif

.DEFAULT_GOAL=all
.PHONY: all
all: $(DEPS) $(EXEC)
	@echo done

.PHONY: clean
clean:
	-rm -f $(OBJS) $(DEPS) $(HDRS:.h=.h.gch) $(EXEC) *.out
	@echo done

.PHONY: debug
debug: $(DEPS) $(EXEC)
	@echo done

.PHONY: depend
depend: $(DEPS)
	@echo done

.PHONY: headers
headers: $(HDRS:.h=.h.gch)
	@echo done

.PHONY: dox
dox: Doxyfile
	doxygen Doxyfile

$(EXEC): $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) -c $(CFLAGS) -o $@ $<

%.d: %.c
	$(CC) $(DFLAGS) $< >$*.d

%.h.gch: %.h
	$(CC) -c $(CFLAGS) -o $@ $<

Doxyfile:
	doxygen -g

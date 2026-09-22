# Build the technical report.
#
# Requires tectonic (https://tectonic-typesetting.github.io), a self-contained
# LaTeX engine that downloads the packages it needs on first run:
#     cargo install tectonic     # or: conda install -c conda-forge tectonic
#
# The figures come from docs/figures/, which are copied out of a run directory;
# regenerate them with the command in Appendix A of the report.

TEX     := report.tex
PDF     := report.pdf
FIGURES := $(wildcard ../docs/figures/*.png)

.PHONY: all clean watch

all: $(PDF)

$(PDF): $(TEX) preamble.tex $(FIGURES)
	tectonic $(TEX)

watch:
	tectonic -X watch

clean:
	rm -f $(PDF) *.aux *.log *.out *.toc

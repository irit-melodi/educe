TUTORIALS=stac pdtb rst-dt
TUTORIALS_HTML=$(patsubst %,%.html,$(TUTORIALS))
GIT_BRANCH=$(shell git rev-parse --abbrev-ref HEAD)

.PHONY: convert clean update

convert: update $(TUTORIALS_HTML)

update:
ifeq ($(GIT_BRANCH),gh-pages)
	git fetch origin master:master
	git checkout master .
endif

clean:
	rm -f $(TUTORIALS_HTML)

%.html: %.ipynb
	ipython nbconvert --to html $(<)

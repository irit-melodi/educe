TUTORIALS=stac pdtb rst-dt
TUTORIALS_HTML=$(patsubst %,%.html,$(TUTORIALS))

.PHONY: convert clean

convert: $(TUTORIALS_HTML)

clean:
	rm -f $(TUTORIALS_HTML)

%.html: %.ipynb
	ipython nbconvert --to html $(<)

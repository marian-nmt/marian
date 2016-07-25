DOMAIN=news
SRC=de
TRG=en

MOSES_SCRIPTS=/opt/moses/moses-scripts
AMUNMT_BIN=../../build/bin

DIRECTION=$(SRC)-$(TRG)
DIR=$(DOMAIN)-$(DIRECTION)

NUMBER_OF_LINES=99999

.SECONDARY:

$(DIR)/bleu.points: $(DIR)/bleu.score
	perl -ne '/BLEU = (\d+\.\d+)/; print "BLEU\n$$1\n"' < $< > $@

$(DIR)/bleu.score: $(DIR)/test-$(SRC)$(TRG)-ref.$(TRG).tok $(DIR)/test-$(SRC)$(TRG)-out.$(TRG).tok
	$(MOSES_SCRIPTS)/generic/multi-bleu.perl $< < $(DIR)/test-$(SRC)$(TRG)-out.$(TRG).tok > $@

%.$(TRG).tok: %.$(TRG).txt
	$(MOSES_SCRIPTS)/tokenizer/tokenizer.perl -l $(TRG) -a < $< > $@

%-out.$(TRG).txt: %-out.$(TRG).bpe
	sed 's/\@\@ //g' $< | \
	$(MOSES_SCRIPTS)/recaser/detruecase.perl | \
	$(MOSES_SCRIPTS)/tokenizer/detokenizer.perl -l $(TRG) > $@

%-out.$(TRG).bpe: %-src.$(SRC).bpe $(DIR)/config.yml
	head -n $(NUMBER_OF_LINES) < $< | $(AMUNMT_BIN)/amun -c $(DIR)/config.yml > $@

%.$(SRC).bpe: %.$(SRC).pre $(DIR)/$(SRC)$(TRG).bpe
	$(AMUNMT_BIN)/bpe $(DIR)/$(SRC)$(TRG).bpe < $< > $@

%.$(SRC).pre: %.$(SRC).txt $(DIR)/truecase-model.$(SRC)
	$(MOSES_SCRIPTS)/tokenizer/normalize-punctuation.perl -l $(SRC) < $< | \
	$(MOSES_SCRIPTS)/tokenizer/tokenizer.perl -l $(SRC) -a | \
	$(MOSES_SCRIPTS)/recaser/truecase.perl -model $(DIR)/truecase-model.$(SRC) > $@


$(DIR)/test-$(SRC)$(TRG)-ref.$(TRG).txt: $(DIR)/test-$(SRC)$(TRG)-ref.$(TRG).sgm
	./extract_segs.py < $< > $@

$(DIR)/test-$(SRC)$(TRG)-src.$(SRC).txt: $(DIR)/test-$(SRC)$(TRG)-src.$(SRC).sgm
	./extract_segs.py < $< > $@

$(DIR)/test-$(SRC)$(TRG)-ref.$(TRG).sgm: $(DIR)/test.tgz
	tar --to-stdout -zxvf $< test/$(DOMAIN)test2016-$(SRC)$(TRG)-ref.$(TRG).sgm > $@

$(DIR)/test-$(SRC)$(TRG)-src.$(SRC).sgm: $(DIR)/test.tgz
	tar --to-stdout -zxvf $< test/$(DOMAIN)test2016-$(SRC)$(TRG)-src.$(SRC).sgm > $@

$(DIR)/test.tgz:
	mkdir -p $(DIR)
	wget 'http://data.statmt.org/wmt16/translation-task/test.tgz' -O $@

$(DIR)/model.npz $(DIR)/vocab.$(SRC).json $(DIR)/vocab.$(TRG).json $(DIR)/$(SRC)$(TRG).bpe $(DIR)/truecase-model.$(SRC) $(DIR)/config.yml: ../../scripts/download_models.py
	mkdir -p $(DIR)
	$< -w $(DIR) -m $(DIRECTION)

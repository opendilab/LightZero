PYTHON := $(shell which python)

SOURCE         ?= .
PYTHON_DEMOS   := $(shell find ${SOURCE} -name *.demo.py)
PYTHON_DEMOXS  := $(shell find ${SOURCE} -name *.demox.py)
PYTHON_RESULTS := $(addsuffix .py.txt, $(basename ${PYTHON_DEMOS} ${PYTHON_DEMOXS}))

SHELL_DEMOS    := $(shell find ${SOURCE} -name *.demo.sh)
SHELL_DEMOXS   := $(shell find ${SOURCE} -name *.demox.sh)
SHELL_RESULTS  := $(addsuffix .sh.txt, $(basename ${SHELL_DEMOS} ${SHELL_DEMOXS}))

%.demo.py.txt: %.demo.py
	cd "$(shell dirname $(shell readlink -f $<))" && \
		PYTHONPATH="$(shell dirname $(shell readlink -f $<)):${PYTHONPATH}" \
		$(PYTHON) "$(shell readlink -f $<)" > "$(shell readlink -f $@)"

%.demox.py.txt: %.demox.py
	cd "$(shell dirname $(shell readlink -f $<))" && \
		PYTHONPATH="$(shell dirname $(shell readlink -f $<)):${PYTHONPATH}" \
		$(PYTHON) "$(shell readlink -f $<)" 1> "$(shell readlink -f $@)" \
		2> "$(shell readlink -f $(addsuffix .err, $(basename $@)))"; \
		echo $$? > "$(shell readlink -f $(addsuffix .exitcode, $(basename $@)))"

%.demo.sh.txt: %.demo.sh
	cd "$(shell dirname $(shell readlink -f $<))" && \
		PYTHONPATH="$(shell dirname $(shell readlink -f $<)):${PYTHONPATH}" \
		$(SHELL) "$(shell readlink -f $<)" > "$(shell readlink -f $@)"

%.demox.sh.txt: %.demox.sh
	cd "$(shell dirname $(shell readlink -f $<))" && \
		PYTHONPATH="$(shell dirname $(shell readlink -f $<)):${PYTHONPATH}" \
		$(SHELL) "$(shell readlink -f $<)" 1> "$(shell readlink -f $@)" \
		2> "$(shell readlink -f $(addsuffix .err, $(basename $@)))"; \
		echo $$? > "$(shell readlink -f $(addsuffix .exitcode, $(basename $@)))"

build: ${PYTHON_RESULTS} ${SHELL_RESULTS}

all: build

clean:
	rm -rf \
		$(shell find ${SOURCE} -name *.py.txt) \
		$(shell find ${SOURCE} -name *.py.err) \
		$(shell find ${SOURCE} -name *.py.exitcode) \
		$(shell find ${SOURCE} -name *.sh.txt) \
		$(shell find ${SOURCE} -name *.sh.err) \
		$(shell find ${SOURCE} -name *.sh.exitcode) \
		$(shell find ${SOURCE} -name *.dat.*)

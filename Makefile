.PHONY: docs test unittest build clean benchmark zip

NO_DEBUG     ?=
NO_DOCSTRING ?=
NO_DEBUG_CMD := $(if ${NO_DOCSTRING},-OO,$(if ${NO_DEBUG},-O,))
PYTHON       ?= $(shell which python) ${NO_DEBUG_CMD}

DOC_DIR        := ./docs
DIST_DIR       := ./dist
WHEELHOUSE_DIR := ./wheelhouse
BENCHMARK_DIR  := ./benchmark
SRC_DIR        := ./lzero
RUNS_DIR       := ./runs

RANGE_DIR       ?= .
RANGE_TEST_DIR  := ${SRC_DIR}/${RANGE_DIR}
RANGE_BENCH_DIR := ${BENCHMARK_DIR}/${RANGE_DIR}
RANGE_SRC_DIR   := ${SRC_DIR}/${RANGE_DIR}

CYTHON_FILES   := $(shell find ${SRC_DIR} -name '*.pyx')
CYTHON_RELATED := \
	$(addsuffix .c, $(basename ${CYTHON_FILES})) \
	$(addsuffix .cpp, $(basename ${CYTHON_FILES})) \
	$(addsuffix .h, $(basename ${CYTHON_FILES})) \

COV_TYPES        ?= xml term-missing
COMPILE_PLATFORM ?= manylinux_2_24_x86_64


build:
	CC=g++ $(PYTHON) setup.py build_ext --inplace \
					$(if ${LINETRACE},--define CYTHON_TRACE,)

zip:
	$(PYTHON) -m build --sdist --outdir ${DIST_DIR}

package:
	$(PYTHON) -m build --sdist --wheel --outdir ${DIST_DIR}
	for whl in `ls ${DIST_DIR}/*.whl`; do \
		auditwheel repair $$whl -w ${WHEELHOUSE_DIR} --plat ${COMPILE_PLATFORM} && \
		cp `ls ${WHEELHOUSE_DIR}/*.whl` ${DIST_DIR} && \
		rm -rf $$whl ${WHEELHOUSE_DIR}/* \
  	; done

clean:
	rm -rf $(shell find ${SRC_DIR} -name '*.so') \
			$(if ${CYTHON_RELATED},$(shell ls ${CYTHON_RELATED} 2> /dev/null),)
	rm -rf ${DIST_DIR} ${WHEELHOUSE_DIR}

test: unittest benchmark

unittest:
	$(PYTHON) -m pytest "${RANGE_TEST_DIR}" \
		-sv -m unittest \
		$(shell for type in ${COV_TYPES}; do echo "--cov-report=$$type"; done) \
		--cov="${RANGE_SRC_DIR}" \
		$(if ${MIN_COVERAGE},--cov-fail-under=${MIN_COVERAGE},) \
		$(if ${WORKERS},-n ${WORKERS},)

minitest:
	$(PYTHON) -m pytest "${SRC_DIR}/mcts/tests/test_game_block.py" \
		-sv -m unittest \
		$(shell for type in ${COV_TYPES}; do echo "--cov-report=$$type"; done) \
		--cov="${SRC_DIR}/mcts/tests/test_game_block.py" \
		$(if ${MIN_COVERAGE},--cov-fail-under=${MIN_COVERAGE},) \
		$(if ${WORKERS},-n ${WORKERS},)

docs:
	$(MAKE) -C "${DOC_DIR}" build
pdocs:
	$(MAKE) -C "${DOC_DIR}" prod

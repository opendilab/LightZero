PLANTUMLCLI ?= $(shell which plantumlcli)

SOURCE ?= .
PUMLS  := $(shell find ${SOURCE} -name *.puml)
PNGS   := $(addsuffix .puml.png, $(basename ${PUMLS}))
SVGS   := $(addsuffix .puml.svg, $(basename ${PUMLS}))

%.puml.png: %.puml
	$(PLANTUMLCLI) -t png -o "$(shell readlink -f $@)" "$(shell readlink -f $<)"

%.puml.svg: %.puml
	$(PLANTUMLCLI) -t svg -o "$(shell readlink -f $@)" "$(shell readlink -f $<)"

build: ${SVGS} ${PNGS}

all: build

clean:
	rm -rf \
		$(shell find ${SOURCE} -name *.puml.svg) \
		$(shell find ${SOURCE} -name *.puml.png) \

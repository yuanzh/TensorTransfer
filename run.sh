#!/bin/sh

lang=$1
runid=$2
shift
shift

java -classpath "bin:lib/trove.jar" -Xmx10000m parser.TensorTransfer target:$lang model-file:runs/$lang.model.$runid train test typo-file:data/universal_treebanks_v2.0/typo.txt out-file:runs/$lang.output.$runid model-file:runs/$lang.model.$runid feature:basic label:true proj:true $@ | tee runs/$lang.$runid.log

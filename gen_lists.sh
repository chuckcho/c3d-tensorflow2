#!/usr/bin/env bash

UCF101PATH=/media/6TB/UCF-101
CATEGORYFILE=lists/categories.lst
CORRECTNUMCATEGORIES=101

NUMCATEGORIES=$(find \
  ${UCF101PATH}/* \
  -maxdepth 1 \
  -name "[A-Z]*" -and ! -name "*_*" \
  -exec basename {} \; \
  | sort | wc -l)

if [ ${NUMCATEGORIES} -ne ${CORRECTNUMCATEGORIES} ]; then
  echo "[Error] NUMCATEGORIES found=${NUMCATEGORIES}: not equal to ${CORRECTNUMCATEGORIES}. Check UCF101 path=${UCF101PATH}, and rerun this script."
  exit -1
fi

echo "Creating ${CATEGORYFILE} with:"
find \
  ${UCF101PATH}/* \
  -maxdepth 1 \
  -name "[A-Z]*" -and ! -name "*_*" \
  -exec basename {} \; \
  | sort | xargs

find \
  ${UCF101PATH}/* \
  -maxdepth 1 \
  -name "[A-Z]*" -and ! -name "*_*" \
  -exec basename {} \; \
  | sort \
  > ${CATEGORYFILE}

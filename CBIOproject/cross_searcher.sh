#!/bin/sh
for filename in `find Data/*.csv`; do 
    echo ${filename};
	head -n 1 ~tommy/ind/hg19.CpG.450K.bed | sed 's/\t/,/g' >! ${filename}_locations.csv;
    for key in `awk -F , '{ if ( NR > 4 ) print $1 }' ${filename}`; do 
        grep ${key} ~tommy/ind/hg19.CpG.450K.bed | sed 's/\t/,/g' >> ${filename}_locations.csv; 
    done; 
done;
